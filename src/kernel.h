#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

using namespace std;


#define VISUALIZE 1
#define CPU false
#define GPU true
#define KDTREE true
#define RANDTRANSFORM true
#define INITTRANSFORM true

namespace Scan_Matching {
    void initSimulation(int N, vector<glm::vec3>& originalPoints, vector<glm::vec3>& transformedPoints, glm::vec4 *kdTree, int kdTreeLength);
	void runCPU(int N, vector<glm::vec3>& originalPoints, vector<glm::vec3>& transformedPoints);
	void runGPU(int numFinalPoints, int numInitialPoints);
	void runGPUWithKDTree(int numFinalPoints, int numInitialPoints, int kdTreeLength);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();

	class Node {
	public:
		__device__ Node(int p, int d, bool g) {
			pointIdx = p;
			depth = d;
			goodNode = g;
		}
		int depth;
		int pointIdx;
		bool goodNode;

	};

	void create(std::vector<glm::vec3> input, glm::vec4 *list, int start, int end, int self, int recursionDepth);
}
