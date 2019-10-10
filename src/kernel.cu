#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include <thrust/reduce.h>
#include "kernel.h"
#include "svd3.h"
#include <time.h>

using namespace std;

#define TIMER false
clock_t timer;
double total_time = 0;

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 256

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_correspond;
glm::vec4 *dev_kdTree;
Scan_Matching::Node *dev_kdTree_stack;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__global__ void kernColorPoints(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}




/**
* Initialize memory, update some globals
*/
void Scan_Matching::initSimulation(int N, vector<glm::vec3>& transformedPoints, vector<glm::vec3>& originalPoints, glm::vec4 *kdTree, int kdTreeLength) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_correspond, originalPoints.size() * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_correspond failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");
  cudaDeviceSynchronize();


  // copy both scene and target to output points
  cudaMemcpy(dev_pos, &originalPoints[0], originalPoints.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("cudaMemcpy failed!");
  cudaMemcpy(&dev_pos[originalPoints.size()], &transformedPoints[0], transformedPoints.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  checkCUDAErrorWithLine("cudaMemcpy failed!");



  if (KDTREE) {
	  cudaMalloc((void**)&dev_kdTree, kdTreeLength * sizeof(glm::vec4));
	  checkCUDAErrorWithLine("cudaMalloc dev_kdTree failed!");

	  cudaMalloc((void**)&dev_kdTree_stack, transformedPoints.size() * ceil(log2(kdTreeLength)) * sizeof(Node));
	  checkCUDAErrorWithLine("cudaMalloc dev_kdTree_stack failed!");

	  cudaMemcpy(dev_kdTree, &kdTree[0], kdTreeLength * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	  checkCUDAErrorWithLine("cudaMemcpy failed!");
  }

  kernColorPoints << <dim3((transformedPoints.size() + blockSize - 1) / blockSize), blockSize >> > (transformedPoints.size(), dev_vel1, glm::vec3(1, 1, 0));
  checkCUDAErrorWithLine("kernColorPoints failed!");
  kernColorPoints << <dim3((originalPoints.size() + blockSize - 1) / blockSize), blockSize >> > (originalPoints.size(), &dev_vel1[transformedPoints.size()], glm::vec3(0, 1, 0));
  checkCUDAErrorWithLine("kernColorPoints failed!");

}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Scan_Matching::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

void Scan_Matching::runCPU(int N, vector<glm::vec3>& finalPoints, vector<glm::vec3>& initialPoints) {
	if (TIMER) {
		timer = clock();
	}
	vector<glm::vec3> correspondingPoints;
	glm::vec3 initialCenter(0.0f, 0.0f, 0.0f);
	glm::vec3 correspondingCenter(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < initialPoints.size(); i++) {
		float minDistance = glm::distance(initialPoints[i], finalPoints[0]);
		glm::vec3 closestPoint = finalPoints[0];
		for (int j = 1; j < finalPoints.size(); j++) {
			if (glm::distance(initialPoints[i], finalPoints[j]) < minDistance) {
				minDistance = glm::distance(initialPoints[i], finalPoints[j]);
				closestPoint = finalPoints[j];
			}
		}
		correspondingPoints.push_back(closestPoint);
		initialCenter = initialCenter + initialPoints[i];
		correspondingCenter = correspondingCenter + closestPoint;
	}

	initialCenter /= initialPoints.size();
	correspondingCenter /= initialPoints.size();
	float W[3][3] = { 0 };
	for (int k = 0; k < initialPoints.size(); k++) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				W[i][j] += ((correspondingPoints[k] - correspondingCenter)[i]) * ((initialPoints[k] - initialCenter)[j]);
			}
		}
	}
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 R = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2])) * 
		glm::mat3(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));
	if (glm::determinant(R) < 0) {
		R = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2])) *
			glm::mat3(glm::vec3(V[0][0], V[0][1], -1.0f * V[0][2]), glm::vec3(V[1][0], V[1][1], -1.0f * V[1][2]), glm::vec3(V[2][0], V[2][1], -1.0f * V[2][2]));
	}
	glm::vec3 T = correspondingCenter - (R * initialCenter);
	for (int i = 0; i < initialPoints.size(); i++) {
		initialPoints[i] = R * initialPoints[i] + T;
	}
	cudaMemcpy(dev_pos, &initialPoints[0], initialPoints.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	if (TIMER) {
		timer = clock() - timer;
		total_time = ((double)timer) / CLOCKS_PER_SEC;
	}
	if (TIMER) {
		printf("(Time for this iteration : %f \n", total_time);
	}
}



__global__ void kernFindCorrespondingPoint(glm::vec3 *initialPoints, int numInitialPoints, glm::vec3 *correspondingPoints, int numFinalPoints) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numInitialPoints) {
		glm::vec3 point = initialPoints[index];
		glm::vec3 closestPoint = initialPoints[numInitialPoints];
		float minDistance = glm::distance(point, closestPoint);
		for (int j = 1; j < numFinalPoints; j++) {
			glm::vec3 newPoint = initialPoints[j + numInitialPoints];
			if (glm::distance(point, newPoint) < minDistance) {
				minDistance = glm::distance(point, newPoint);
				closestPoint = newPoint;
			}
		}
		correspondingPoints[index] = closestPoint;
	}
}

__global__ void kernTraverseKDTree(int numInitialPoints, int numOfCol, int kdTreeLength, glm::vec3 *dev_pos, Scan_Matching::Node *dev_kdTree_stack, glm::vec4* dev_kdTree, glm::vec3 *correspondingPoints) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numInitialPoints) {
		int top = 0;
		float minDistance = FLT_MAX;
		glm::vec3 closestPoint;
		Scan_Matching::Node startNode(0,0,true);
		dev_kdTree_stack[index * numOfCol + top] = startNode;
		glm::vec3 basePoint = dev_pos[index];
		while (top >= 0) {
			Scan_Matching::Node currentNode = dev_kdTree_stack[index * numOfCol + top--];
			if (dev_kdTree[currentNode.pointIdx][3] == 0.0f) {
				continue;
			}
			glm::vec3 newPoint = glm::vec3(dev_kdTree[currentNode.pointIdx]);
			int axis = currentNode.depth - (3 * (int)(currentNode.depth / 3));
			if (!currentNode.goodNode) {
				glm::vec3 newPointParent = glm::vec3(dev_kdTree[(currentNode.pointIdx - 1)/2]);
				if (abs(basePoint[axis] - newPointParent[axis]) >= minDistance) {
					continue;
				}
			}

			float newDistance = glm::distance(newPoint, basePoint);
			if (newDistance < minDistance) {
				closestPoint = newPoint;
				minDistance = newDistance;
			}

			bool left = basePoint[axis] < newPoint[axis];

			int goodSide = (2 * currentNode.pointIdx) + ((left) ? 1 : 2);
			int badSide = (2 * currentNode.pointIdx) + ((left) ? 2 : 1);

			if (badSide < kdTreeLength) {

				Scan_Matching::Node badNode(badSide, currentNode.depth + 1, false);
				dev_kdTree_stack[index * numOfCol + ++top] = badNode;
			}
			if (goodSide < kdTreeLength) {

				Scan_Matching::Node goodNode(goodSide, currentNode.depth + 1, true);
				dev_kdTree_stack[index * numOfCol + ++top] = goodNode;
			}
		}
		correspondingPoints[index] = closestPoint;
	}
}

__global__ void kernMultMatrices(glm::vec3 *initialPoints, int numInitialPoints, glm::vec3 *correspondingPoints, glm::vec3 initialCenter, glm::vec3 correspondingCenter,glm::mat3 *prod) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < numInitialPoints)
		prod[index] = glm::outerProduct(initialPoints[index] - initialCenter, correspondingPoints[index] - correspondingCenter);
}

__global__ void kernUpdatePoints(glm::vec3 *initialPoints, int numInitialPoints, glm::mat3 R, glm::vec3 T){
int index = (blockIdx.x * blockDim.x) + threadIdx.x;
if (index < numInitialPoints)
	initialPoints[index] = R * initialPoints[index] + T;
}


void Scan_Matching::runGPU(int numFinalPoints, int numInitialPoints) {
	cout << "At Line : " << __LINE__ << endl;
	if (TIMER) {
		timer = clock();
	}
	dim3 fullBlocksPerGrid((numInitialPoints + blockSize - 1) / blockSize);
	vector<glm::vec3> correspondingPoints;

	kernFindCorrespondingPoint << <fullBlocksPerGrid, blockSize >> > (dev_pos, numInitialPoints, dev_correspond, numFinalPoints);
	checkCUDAErrorWithLine("kernFindCorrespondingPoint failed!");

	thrust::device_ptr<glm::vec3> thrust_dev_pos(dev_pos);
	thrust::device_ptr<glm::vec3> thrust_dev_correspond(dev_correspond);

	glm::vec3 initialCenter = glm::vec3(thrust::reduce(thrust_dev_pos, thrust_dev_pos + numInitialPoints, glm::vec3(0.0f, 0.0f, 0.0f)));
	glm::vec3 correspondingCenter = glm::vec3(thrust::reduce(thrust_dev_correspond, thrust_dev_correspond + numInitialPoints, glm::vec3(0.0f, 0.0f, 0.0f)));

	initialCenter /= numInitialPoints;
	correspondingCenter /= numInitialPoints;

	glm::mat3 *dev_prod;

	cudaMalloc((void**)&dev_prod, numInitialPoints * sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMalloc dev_prod failed!");

	kernMultMatrices << <fullBlocksPerGrid, blockSize >> > (dev_pos, numInitialPoints,dev_correspond, initialCenter, correspondingCenter, dev_prod);
	checkCUDAErrorWithLine("kernMultMatrices failed!");
	glm::mat3 W = thrust::reduce(thrust::device, dev_prod, dev_prod + numInitialPoints, glm::mat3(0));

	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 R = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2])) *
		glm::mat3(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));
	if (glm::determinant(R) < 0) {
		R = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2])) *
			glm::mat3(glm::vec3(V[0][0], V[0][1], -1.0f * V[0][2]), glm::vec3(V[1][0], V[1][1], -1.0f * V[1][2]), glm::vec3(V[2][0], V[2][1], -1.0f * V[2][2]));
	}
	glm::vec3 T = correspondingCenter - (R * initialCenter);

	kernUpdatePoints << <fullBlocksPerGrid, blockSize >> > (dev_pos, numInitialPoints, R, T);
	checkCUDAErrorWithLine("kernUpdatePoints failed!");
	cudaDeviceSynchronize();

	if (TIMER) {
		timer = clock() - timer;
		total_time = ((double)timer) / CLOCKS_PER_SEC;
	}
	if (TIMER) {
		printf("(Time for this iteration : %f \n", total_time);
	}
}

void Scan_Matching::runGPUWithKDTree(int numFinalPoints, int numInitialPoints, int kdTreeLength) {

	dim3 fullBlocksPerGrid((numInitialPoints + blockSize - 1) / blockSize);

	//kernFindCorrespondingPoint << <fullBlocksPerGrid, blockSize >> > (dev_pos, numInitialPoints, dev_correspond, numFinalPoints);
	kernTraverseKDTree << <fullBlocksPerGrid, blockSize >> > (numInitialPoints, ceil(log2(kdTreeLength)), kdTreeLength, dev_pos, dev_kdTree_stack, dev_kdTree, dev_correspond);
	checkCUDAErrorWithLine("kernTraverseKDTree failed!");

	thrust::device_ptr<glm::vec3> thrust_dev_pos(dev_pos);
	thrust::device_ptr<glm::vec3> thrust_dev_correspond(dev_correspond);

	glm::vec3 initialCenter = glm::vec3(thrust::reduce(thrust_dev_pos, thrust_dev_pos + numInitialPoints, glm::vec3(0.0f, 0.0f, 0.0f)));
	glm::vec3 correspondingCenter = glm::vec3(thrust::reduce(thrust_dev_correspond, thrust_dev_correspond + numInitialPoints, glm::vec3(0.0f, 0.0f, 0.0f)));

	initialCenter /= numInitialPoints;
	correspondingCenter /= numInitialPoints;

	glm::mat3 *dev_prod;

	cudaMalloc((void**)&dev_prod, numInitialPoints * sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMalloc dev_prod failed!");

	kernMultMatrices << <fullBlocksPerGrid, blockSize >> > (dev_pos, numInitialPoints, dev_correspond, initialCenter, correspondingCenter, dev_prod);
	checkCUDAErrorWithLine("kernMultMatrices failed!");
	glm::mat3 W = thrust::reduce(thrust::device, dev_prod, dev_prod + numInitialPoints, glm::mat3(0));

	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };
	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
	);

	glm::mat3 R = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2])) *
		glm::mat3(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));
	if (glm::determinant(R) < 0) {
		R = glm::mat3(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2])) *
			glm::mat3(glm::vec3(V[0][0], V[0][1], -1.0f * V[0][2]), glm::vec3(V[1][0], V[1][1], -1.0f * V[1][2]), glm::vec3(V[2][0], V[2][1], -1.0f * V[2][2]));
	}
	glm::vec3 T = correspondingCenter - (R * initialCenter);

	kernUpdatePoints << <fullBlocksPerGrid, blockSize >> > (dev_pos, numInitialPoints, R, T);
	checkCUDAErrorWithLine("kernUpdatePoints failed!");
	cudaDeviceSynchronize();

	if (TIMER) {
		timer = clock() - timer;
		total_time = ((double)timer) / CLOCKS_PER_SEC;
	}
	if (TIMER) {
		printf("(Time for this iteration : %f \n", total_time);
	}
}

void Scan_Matching::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);



  if (KDTREE) {
	  cudaFree(dev_kdTree);
	  cudaFree(dev_kdTree_stack);
	  cudaFree(dev_kdTree);
  }

}
bool sortX(const glm::vec3 &p1, const glm::vec3 &p2)
{
	return p1.x < p2.x;
}
bool sortY(const glm::vec3 &p1, const glm::vec3 &p2)
{
	return p1.y < p2.y;
}
bool sortZ(const glm::vec3 &p1, const glm::vec3 &p2)
{
	return p1.z < p2.z;
}

//__device__ Scan_Matching::Node::Node(int  p, int d, bool g) {
//	pointIdx = p;
//	depth = d;
//	goodNode = g;
//}

void Scan_Matching::create(std::vector<glm::vec3> input, glm::vec4 *list, int start, int end, int self, int recursionDepth) {

	if (start > end)
		return;

	if (recursionDepth % 3 == 0) {
		sort(input.begin() + start, input.begin() + end + 1, sortX);
	}

	if (recursionDepth % 3 == 1) {
		sort(input.begin() + start, input.begin() + end + 1, sortY);
	}

	if (recursionDepth % 3 == 2) {
		sort(input.begin() + start, input.begin() + end + 1, sortZ);
	}

	// set current node
	int mid = (int)((start + end) / 2);
	list[self] = glm::vec4(input[mid].x, input[mid].y, input[mid].z, 1.0f);
	create(input, list, start, mid - 1, 2 * self + 1, recursionDepth + 1);
	create(input, list, mid + 1, end, 2 * self + 2, recursionDepth + 1);
}