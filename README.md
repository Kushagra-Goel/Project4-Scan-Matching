**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Author : Kushagra
  * [LinkedIn](https://www.linkedin.com/in/kushagragoel/)
* Tested on : Windows 10, i7-9750H CPU @ 2.60GHz 16GB, GTX 1650 4GB (Personal Computer)

____________________________________________________________________________________

# Boids

## What are Boids?
Boids is an artificial life program developed by Craig W. Reynolds which simulates a very common flocking behavior found in nature among animals like birds and fishes. A boid shows an emergent behavior which is directly dependent on its neighbors. Its behavior arises from 3 main simple rules :

* **separation**: steer to avoid crowding local flockmates

* **alignment**: steer towards the average heading of local flockmates

* **cohesion**: steer to move towards the average position (center of mass) of local flockmates

Other rules can be added like obstacle avoidance and goal seeking. In this project we simulate the Boids Flocking with 3 different implementations :

* **Naive** : Each boid checks all other boids to calculate the behavior.
* **Uniform Grid** : Each boid checks all other boids in surrounding cells to calculate the behavior.
* **Uniform Grid with coherent memory access **: Each boid checks all other boids in surrounding cells to calculate the behavior. Here we reshuffle the boids to remove extra memory accesses.

Example visuals of Uniform grid with coherent memory access, the most efficient implementation.

| 5,000 Boids | 50,000 Boids |
| ------------- | ----------- |
| ![](img/Coherent5000.gif)  | ![](img/Coherent50000.gif) |

## Rules
In the Boids flocking simulation, particles representing birds or fish (boids) move around the simulation space according to three rules:

* Cohesion - boids move towards the perceived center of mass of their neighbors

* Separation - boids avoid getting to close to their neighbors

* Alignment - boids generally try to move with the same direction and speed as
  their neighbors

These three rules specify a boid's velocity change in a timestep.
At every timestep, a boid thus has to look at each of its neighboring boids
and compute the velocity change contribution from each of the three rules.
Thus, a bare-bones boids implementation has each boid check every other boid in
the simulation.

#### Rule 1: Boids try to fly towards the centre of mass of neighbouring boids

```
function rule1(Boid boid)

    Vector perceived_center

    foreach Boid b:
        if b != boid and distance(b, boid) < rule1Distance then
            perceived_center += b.position
        endif
    end

    perceived_center /= number_of_neighbors

    return (perceived_center - boid.position) * rule1Scale
end
```

#### Rule 2: Boids try to keep a small distance away from other objects (including other boids).

```
function rule2(Boid boid)

    Vector c = 0

    foreach Boid b
        if b != boid and distance(b, boid) < rule2Distance then
            c -= (b.position - boid.position)
        endif
    end

    return c * rule2Scale
end
```

#### Rule 3: Boids try to match velocity with near boids.

```
function rule3(Boid boid)

    Vector perceived_velocity

    foreach Boid b
        if b != boid and distance(b, boid) < rule3Distance then
            perceived_velocity += b.velocity
        endif
    end

    perceived_velocity /= number_of_neighbors

    return perceived_velocity * rule3Scale
end
```
Based on [Conard Parker's notes](http://www.vergenet.net/~conrad/boids/pseudocode.html) with slight adaptations. For the purposes of an interesting simulation,
we will say that two boids only influence each other according if they are
within a certain **neighborhood distance** of each other.

Since these only affect the delta by which a current velocity should be updated, the updated velocity for a current value is as follows: `updated_vel = current_vel + rule1_adhesion + rule2_avoidance_dodging + rule3_cohesion

## Part 1: Boids with Naive Neighbor Search

We simply check each boid against every other boid to see if they are within a certain neighborhood distance of each other. 

## Part 2: Let there be (better) flocking!

### A quick explanation of uniform grids

From Part 1, we observe that any two boids can only influence each other if they are
within some *neighborhood distance* of each other.
Based on this observation, we can see that having each boid check every
other boid is very inefficient, especially if (as in our standard parameters)
the number of boids is large and the neighborhood distance is much smaller than
the full simulation space. We can cull a lot of neighbor checks using a
datastructure called a **uniform spatial grid**.

A uniform grid is made up of cells that are at least as wide as the neighborhood
distance and covers the entire simulation domain.
Before computing the new velocities of the boids, we "bin" them into the grid in
a preprocess step.
![a uniform grid in 2D](images/Boids%20Ugrid%20base.png)

If the cell width is double the neighborhood distance, each boid only has to be
checked against other boids in 8 cells, or 4 in the 2D case.

![a uniform grid in 2D with neighborhood and cells to search for some particles shown](images/Boids%20Ugrid%20neighbor%20search%20shown.png)

We can build a uniform grid on the CPU by iterating over the boids, figuring out
its enclosing cell, and then keeping a pointer to the boid in a resizeable
array representing the cell. However, this doesn't transfer well to the GPU
because:

1. We don't have resizeable arrays on the GPU
2. Naively parallelizing the iteration may lead to race conditions, where two
particles need to be written into the same bucket on the same clock cycle.

Instead, we will construct the uniform grid by sorting. If we label each boid
with an index representing its enclosing cell and then sort the list of
boids by these indices, we can ensure that pointers to boids in the same cells
are contiguous in memory.

Then, we can walk over the array of sorted uniform grid indices and look at
every pair of values. If the values differ, we know that we are at the border
of the representation of two different cells. Storing these locations in a table
with an entry for each cell gives us a complete representation of the uniform
grid. This "table" can just be an array with as much space as there are cells.
This process is data parallel and can be naively parallelized.
![buffers for generating a uniform grid using index sort](images/Boids%20Ugrids%20buffers%20naive.png)

## Part 3 Cutting out the middleman : Coherent Memory Access
Consider the uniform grid neighbor search ,pointers to boids in
a single cell are contiguous in memory, but the boid data itself (velocities and
positions) is scattered all over the place. So we rearrange the boid data
itself so that all the velocities and positions of boids in one cell are also
contiguous in memory.

![buffers for generating a uniform grid using index sort, then making the boid data coherent](images/Boids%20Ugrids%20buffers%20data%20coherent.png)


## Additional Optimizations

**Spherical Optimization**

We implemented additional optimizations based on the fact that the neighborhood of a boid is essentially a sphere of the appropriate radius. But by checking all the cells around it wastes computation on the cells for which no part of the sphere is inside the cell. We can eliminate some of these cells by computing the distance of their centroid from the boid and see if they are within the (radius) + (half of the diagonal) distance of each other.

**Grid-Looping Optimization**

Another optimization we implemented was allowing variable cell-width. The system automatically computes what all cells does it need to check in order to get all the neighbors. This is done by calculating the enclosing cube of the sphere (whose radius will be the max of all 3 rule radii) that defines the boundary of neighborhood. Then it calculates what all cells have a part of that approximate cube and then finds all the neighbors.

# Runtime Analysis

**Abbreviations**

* V : Visualize is 1

* NV : Visualize is 0

* UG : Uniform Grid

* UGS : Uniform Grid with Spherical Optimization

* Coherent : Uniform Grid with coherent memory access

* CoherentS : Uniform Grid with coherent memory access and Spherical Optimization

* **Effect of Number of Boids on Frames Per Second**

  * **For each implementation, how does changing the number of boids affect performance? Why do you think this is?**
      * We observe that the number of frames go down as Number of boids increases. This is due to the extra computation that comes with each new boid. Since the density increases, the number of boids in neighborhood of a boid also increase. 
  

<img src="img/FPSvNumBoids.png" alt="NumBoidsVsFPS" width="600"/>  

* **Effect of Block Size of Frames Per Second**

  * **For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**
  * We observe that Block Size effects the FPS negligibly. Although degraded performance was observed for BlockSizes that were not multiple of 32. This might be due to the fact that the warp size is 32, so for any warp that is not started with all 32 threads engaged, the performance is lower compared to the warps that have 100% usage.
  

<img src="img/FPSvBlockSize.png" alt="BlockSizeVsFPS" width="600"/>  

* **Effect of Cell Width on Frames Per Second**

  * **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**
  
    * The effect of Cell Width was tested with 50,000 boids. We see an interesting peak when cell width is exactly the size of the maximum rule radii. This is possibly due to the fact we have a lot of boids and for case where the cell width is greater than 1, the volume covered increases and therefore increasing the number of boids to be checked if they are in the neighborhood. Thus with bigger cell width, more and more boids that are outside the neighborhood sphere increases therefore leading to lower overall performance.
  
  <img src="img/FPSvCellWidth.png" alt="CellWidthVsFPS" width="600"/>  
  
* **For the coherent uniform grid: did you experience any performance improvements**
  **with the more coherent uniform grid? Was this the outcome you expected?**
  **Why or why not?**

  * The performance depended on number of boids. For lower number of boids (in tens and hundreds) the Uniform Grid actually performs better but for high number of boids ( in hundred thousands) the more coherent uniform grid was massively faster than normal uniform grid. This was expected because the reshuffling in coherent access incurs several overheads compared to simple uniform grid. But when we have several thousands of boids, the improvement in latency due coherent memory access and time saved by not waiting for memory fetch allows coherent memory to have higher computation throughput compared to normal uniform grid.