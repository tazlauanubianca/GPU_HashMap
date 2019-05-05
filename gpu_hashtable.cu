#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

#define BLOCKSIZE 1024
#define BLOCKNUM 1
#define A 113
#define B 

typedef struct {
	int state; // 0 -> unoccupied | 1 -> occupied | -1 -> was previously
	int key;
	int value;
} Pair;

typedef struct {
	int *size;
	int *numElem;
	Pair *pairs;
} HashTable;

HashTable hashTable;

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	cudaMalloc(&hashTable.size, sizeof(int));
	cudaMalloc(&hashTable.pairs, size * sizeof(Pair));
	cudaMalloc(&hashTable.numElem, sizeof(int));

	cudaMemset(hashTable.numElem, 0, sizeof(int));
	cudaMemset(hashTable.pairs, 0, size * sizeof(Pair));
	cudaMemcpy(hashTable.size, &size, sizeof(int), cudaMemcpyHostToDevice);
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashTable.pairs);
	cudaFree(hashTable.size);
	cudaFree(hashTable.numElem);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Pair *devicePairs;
	int hostNumElem;
	int hostOldSize;
	int *hostKeys, *hostValues;
	Pair *hostPairs;
	
	//TODO: check the allocations

	/* Get old size of hashTable */
	cudaMemcpy(&hostOldSize, hashTable.size, sizeof(int), cudaMemcpyDeviceToHost);

	/* Get current number of elements */
	cudaMemcpy(&hostNumElem, hashTable.numElem, sizeof(int), cudaMemcpyDeviceToHost);

	/* Get all pairs in hashTable */
	hostPairs = (Pair *) malloc(sizeof(Pair) * hostOldSize);
	cudaMemcpy(hostPairs, hashTable.pairs, sizeof(Pair) * hostOldSize, cudaMemcpyDeviceToHost);

	hostKeys = (int *) malloc(sizeof(int) * hostOldSize);
	hostValues = (int *) malloc(sizeof(int) * hostOldSize);

	/* Filter pairs*/
	int index = 0;
	for (int i = 0; i < hostOldSize; i++) {
		if (hostPairs[i].state == 1) {
			hostKeys[index] = hostPairs[i].key;
			hostValues[index] = hostPairs[i].value;
			index++;
		}
	}

	cudaMalloc(&devicePairs, numBucketsReshape * sizeof(Pair));
	cudaMemset(hashTable.numElem, 0, sizeof(int));
	cudaMemcpy(hashTable.size, &numBucketsReshape, sizeof(int), cudaMemcpyHostToDevice);
	cudaFree(hashTable.pairs);
	
	hashTable.pairs = devicePairs;

	insertBatch(hostKeys, hostValues, hostNumElem);

	free(hostPairs);
	free(hostValues);
	free(hostKeys);
}

__global__ void insert(int *keys, int *values, Pair *pairs, int *size,
		int *numElem, int *result, int numKeys) {
	/* Get position in HashTable for inertion */
	int keyToInsert = blockIdx.x * blockDim.x + threadIdx.x;
	if (keyToInsert >= numKeys)
		return;

	int position = keys[keyToInsert] * A % (*size);
	int index = position;
	int prevOccupied = -1;
	int free = 0;
	int occupied = 1;

	/* Check for an empty space in the HashTable */
	while ((atomicCAS(&(pairs[index].state), prevOccupied, occupied) != -1) &&
		(atomicCAS(&(pairs[index].state), free, occupied) != 0)) {
		index++;
		if (index == (*size))
			index = 0;

		if (index == position)
			return;
	}

	pairs[position].key = keys[keyToInsert];
	pairs[position].value = values[keyToInsert];

	atomicAdd(result, 1);
	atomicAdd(numElem, 1);
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceResult, *deviceKeys, *deviceValues;
	int *hostResult;
	bool returnValue = false;
	
	//TODO: check the allocations
	hostResult = (int *) malloc(sizeof(int));
	*hostResult = 0;

	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMalloc(&deviceResult, sizeof(int));

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceResult, hostResult, sizeof(int), cudaMemcpyHostToDevice);
	
	insert<<<BLOCKNUM, BLOCKSIZE>>>(deviceKeys, deviceValues, hashTable.pairs, 
					hashTable.size, hashTable.numElem, deviceResult, numKeys);
	
	cudaDeviceSynchronize();
	cudaMemcpy(hostResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);
	
	/* Check for result */
	if (*hostResult == numKeys)
		returnValue = true;

	cudaFree(deviceResult);
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	free(hostResult);

	return returnValue;
}

__global__ void get(int *keys, int *values, Pair *pairs, int *size, int numKeys) {
	/* Get position in HashTable for inertion */
	int keyToGet = blockIdx.x * blockDim.x + threadIdx.x;
	if (keyToGet >= numKeys)
		return;

	int position = keys[keyToGet] * A % (*size);
	int key = keys[keyToGet];
	int index = position;

	/* Check for an empty space in the HashTable */
	while (atomicCAS(&(pairs[index].key), key, key) != key) {
		index++;
		if (index == (*size))
			index = 0;

		if (index == position)
			return;
	}

	values[keyToGet] = pairs[index].value;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys, *deviceValues;
	int *hostValues;

	//TODO: check the allocations
	hostValues = (int *) calloc(numKeys, sizeof(int));

	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	get<<<BLOCKNUM, BLOCKSIZE>>>(deviceKeys, deviceValues, hashTable.pairs, 
					hashTable.size, numKeys);
	
	cudaDeviceSynchronize();
	cudaMemcpy(hostValues, deviceValues, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);
	
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	
	return hostValues;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	int *numElem = (int *) malloc(sizeof(int));
	int *size = (int *) malloc(sizeof(int));

	cudaMemcpy(numElem, hashTable.numElem, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(size, hashTable.size, sizeof(int), cudaMemcpyDeviceToHost);
	
	float loadFactor = *numElem / *size;
	
	return loadFactor; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
