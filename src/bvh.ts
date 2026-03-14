import { type Vec3, vec3Add, vec3Max, vec3Min, vec3MulScalar, vec3Sub } from "./math";
import type { MergedGeometry } from "./mesh";

const MIN_VALUE = -1e30;
const MAX_VALUE = 1e30;

export class AABB {
  minCorner: Vec3;
  maxCorner: Vec3;

  constructor(corner1?: Vec3, corner2?: Vec3) {
    if (!corner1 || !corner2) {
      this.minCorner = [MAX_VALUE, MAX_VALUE, MAX_VALUE];
      this.maxCorner = [MIN_VALUE, MIN_VALUE, MIN_VALUE];
    } else {
      this.minCorner = vec3Max(corner1, corner2);
      this.maxCorner = vec3Max(corner1, corner2);
    }
  }

  grow(point: Vec3) {
    this.minCorner = vec3Min(this.minCorner, point);
    this.maxCorner = vec3Max(this.maxCorner, point);
  }

  growToPrimitive(prim: BVHPrimitive) {
    this.grow(prim.v0);
    this.grow(prim.v1);
    this.grow(prim.v2);
  }

  area(): number {
    const diag = vec3Sub(this.maxCorner, this.minCorner);
    return diag[0] * diag[1] + diag[1] * diag[2] + diag[2] * diag[0];
  }

  largestAxis(): number {
    const diag = vec3Sub(this.maxCorner, this.minCorner);

    let largest = 0;
    if (diag[1] > diag[largest]) largest = 1;
    if (diag[2] > diag[largest]) largest = 2;

    return largest;
  }

  exportAABB(): Float32Array<ArrayBuffer> {
    return new Float32Array([...this.minCorner, ...this.maxCorner]);
  }

  static fromVertices(verts: Float32Array): AABB {
    let min: Vec3 = [MAX_VALUE, MAX_VALUE, MAX_VALUE];
    let max: Vec3 = [MIN_VALUE, MIN_VALUE, MIN_VALUE];

    for (let i = 0; i + 2 < verts.length; i += 3) {
      const p: Vec3 = [verts[i], verts[i + 1], verts[i + 2]];

      min = vec3Min(min, p);
      max = vec3Max(max, p);
    }

    return new AABB(min, max);
  }
}

interface SAHBin {
  bounds: AABB;
  numPrimitives: number;
}

class BVHPrimitive {
  index: number; // original index in triangle indices array
  v0: Vec3;
  v1: Vec3;
  v2: Vec3;
  centroid: Vec3;

  constructor(index: number, v0: Vec3, v1: Vec3, v2: Vec3) {
    this.index = index;
    this.v0 = v0;
    this.v1 = v1;
    this.v2 = v2;
    this.centroid = vec3MulScalar(vec3Add(v0, vec3Add(v1, v2)), 1.0 / 3.0);
  }
}

class BVHNode {
  bounds: AABB | null; // null if not calculated yet
  leftChild: number | null; // null if leaf
  rightChild: number | null; // null if leaf

  numPrimitives: number | null; // null if interior
  firstPrimitiveOffset: number | null; // null if interior
  skipLink: number = 0;

  constructor(
    left: number | null, right: number | null,
    numPrimitives: number | null, primitiveOffset: number | null
  ) {
    this.bounds = null;
    this.leftChild = left;
    this.rightChild = right;
    this.numPrimitives = numPrimitives;
    this.firstPrimitiveOffset = primitiveOffset;
  }
}

type BVHHeuristic = "MIDPOINT" | "SAH";

export class BVHTree {
  readonly N_BINS: number = 10;
  readonly rootIdx: number = 0;
  heuristic: BVHHeuristic;
  nodes: BVHNode[];
  size: number;

  triOffset: number;
  primitives: BVHPrimitive[];

  constructor(mesh: MergedGeometry, triOffset: number, triCount: number, vertexOffset: number, heuristic: BVHHeuristic = "SAH") {
    this.size = 0; 
    
    this.heuristic = heuristic;
    this.triOffset = triOffset;
    
    this.primitives = new Array(triCount);
    this.nodes = new Array(2 * triCount - 1);

    for (let i = 0; i < triCount; ++i) {
      const baseIdx = (triOffset + i) * 3;

      const i0 = (mesh.indices[baseIdx] + vertexOffset) * 3;
      const i1 = (mesh.indices[baseIdx + 1] + vertexOffset) * 3;
      const i2 = (mesh.indices[baseIdx + 2] + vertexOffset) * 3;

      const v0: Vec3 = [mesh.positions[i0], mesh.positions[i0 + 1], mesh.positions[i0 + 2]];
      const v1: Vec3 = [mesh.positions[i1], mesh.positions[i1 + 1], mesh.positions[i1 + 2]];
      const v2: Vec3 = [mesh.positions[i2], mesh.positions[i2 + 1], mesh.positions[i2 + 2]];

      this.primitives[i] = new BVHPrimitive(baseIdx, v0, v1, v2);
    }

    this.createLeaf(triCount, 0);
    this.updateBounds(this.rootIdx);
  }

  buildRecursive(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    if (node.numPrimitives! <= 2) return;

    let splitAxis = 0;
    let split = 0;
    if (this.heuristic === "SAH") {
      const [axis, sp, cost] = this.findBestSplit(nodeIdx);
      splitAxis = axis;
      split = sp;

      const noSplitCost = node.numPrimitives! * node.bounds!.area();
      if (cost >= noSplitCost) return;
    } else if (this.heuristic === "MIDPOINT") {
      splitAxis = node.bounds!.largestAxis();

      const midpoint = vec3MulScalar(vec3Add(node.bounds!.minCorner, node.bounds!.maxCorner), 0.5);
      split = midpoint[splitAxis];
    }

    let i = node.firstPrimitiveOffset!;
    let j = i + node.numPrimitives! - 1;
    while (i <= j) {
      if (this.primitives[i].centroid[splitAxis] >= split) {
        const temp = this.primitives[i];
        this.primitives[i] = this.primitives[j];
        this.primitives[j] = temp;
        j--;
      } else {
        i++;
      }
    }

    let nLeft = i - node.firstPrimitiveOffset!;
    if (nLeft === 0 || nLeft === node.numPrimitives!) {
      const first = node.firstPrimitiveOffset!;
      const count = node.numPrimitives!;
      const last = first + count;
      
      const subArray = this.primitives.slice(first, last);
      subArray.sort((a, b) => a.centroid[splitAxis] - b.centroid[splitAxis]);
      
      for (let k = 0; k < count; k++) {
        this.primitives[first + k] = subArray[k];
      }
      
      nLeft = Math.floor(count / 2);
      i = first + nLeft;
    }

    const left = this.createLeaf(nLeft, node.firstPrimitiveOffset!);
    const right = this.createLeaf(node.numPrimitives! - nLeft, i);

    node.numPrimitives = null;
    node.firstPrimitiveOffset = null;
    node.leftChild = left;
    node.rightChild = right;

    this.updateBounds(left);
    this.updateBounds(right);

    this.buildRecursive(left);
    this.buildRecursive(right);
  }

  private findBestSplit(nodeIdx: number): [number, number, number] {
    const node = this.nodes[nodeIdx];
    let splitAxis = -1, split = -1, minCost = MAX_VALUE;

    for (let axis = 0; axis < 3; ++axis) {
      let minBound = MAX_VALUE, maxBound = MIN_VALUE;

      // define start and end bound of the bins
      for (let i = 0; i < node.numPrimitives!; ++i) {
        const tri = node.firstPrimitiveOffset! + i;
        const c = this.primitives[tri].centroid[axis];

        minBound = Math.min(minBound, c);
        maxBound = Math.max(maxBound, c);
      }

      if (minBound === maxBound) continue;

      // fill bins with primitives
      const bins: SAHBin[] = Array.from({ length: this.N_BINS }, () => {
        return { bounds: new AABB(), numPrimitives: 0 };
      });

      let step = this.N_BINS / (maxBound - minBound);
      for (let i = 0; i < node.numPrimitives!; ++i) {
        const tri = node.firstPrimitiveOffset! + i;
        const prim = this.primitives[tri];

        let binIdx = Math.trunc((prim.centroid[axis] - minBound) * step);
        binIdx = Math.min(binIdx, this.N_BINS - 1);

        bins[binIdx].numPrimitives++;
        bins[binIdx].bounds.growToPrimitive(prim);
      }

      // cumulative sum of areas and primitive counts
      const areaLeft: number[] = new Array(this.N_BINS), areaRight: number[] = new Array(this.N_BINS);
      const numLeft: number[] = new Array(this.N_BINS), numRight: number[] = new Array(this.N_BINS);
      let sumLeft = 0, sumRight = 0;
      const leftBox = new AABB(), rightBox = new AABB();
      for (let i = 0; i < this.N_BINS - 1; ++i) {
        sumLeft += bins[i].numPrimitives;
        sumRight += bins[this.N_BINS - 1 - i].numPrimitives;

        numLeft[i] = sumLeft;
        numRight[this.N_BINS - 2 - i] = sumRight;

        if (bins[i].numPrimitives > 0) {
          leftBox.grow(bins[i].bounds.minCorner);
          leftBox.grow(bins[i].bounds.maxCorner);
        }
        
        if (bins[this.N_BINS - 1 - i].numPrimitives > 0) {
          rightBox.grow(bins[this.N_BINS - 1 - i].bounds.minCorner);
          rightBox.grow(bins[this.N_BINS - 1 - i].bounds.maxCorner);
        }

        areaLeft[i] = leftBox.area();
        areaRight[this.N_BINS - 2 - i] = rightBox.area();
      }

      step = (maxBound - minBound) / this.N_BINS;
      for (let i = 0; i < this.N_BINS - 1; ++i) {
        const cost = numLeft[i] * areaLeft[i] + numRight[i] * areaRight[i];
        if (cost < minCost) {
          minCost = cost;
          splitAxis = axis;
          split = minBound + step * (i + 1);
        }
      }
    }

    return [splitAxis, split, minCost];
  }

  private updateBounds(nodeId: number) {
    if (this.nodes[nodeId].numPrimitives === null) throw new Error("Cannot update interior bounds.");

    const first = this.nodes[nodeId].firstPrimitiveOffset!;
    const last = first + this.nodes[nodeId].numPrimitives!;
    const bound = new AABB();

    for (let i = first; i < last; ++i) {
      bound.grow(this.primitives[i].v0);
      bound.grow(this.primitives[i].v1);
      bound.grow(this.primitives[i].v2);
    }

    this.nodes[nodeId].bounds = bound;
  }

  private createLeaf(numPrimitives: number, primitiveOffset: number): number {
    this.nodes[this.size] = new BVHNode(null, null, numPrimitives, primitiveOffset);

    return this.size++;
  }

  reorderIndices(originalIndices: Uint32Array) {
    const newIndices = new Uint32Array(this.primitives.length * 3);
    for (let i = 0; i < this.primitives.length; ++i) {
      const oldBase = this.primitives[i].index; 
      newIndices[i * 3 + 0] = originalIndices[oldBase + 0];
      newIndices[i * 3 + 1] = originalIndices[oldBase + 1];
      newIndices[i * 3 + 2] = originalIndices[oldBase + 2];
    }
    for (let i = 0; i < newIndices.length; ++i) {
      originalIndices[this.triOffset * 3 + i] = newIndices[i];
    }
  }

  flatten(globalOffset: number): BVHNode[] {
    const flat: BVHNode[] = [];

    const traverse = (nodeIdx: number) => {
      const node = this.nodes[nodeIdx];
      flat.push(node);

      if (node.numPrimitives !== null) {
        // leaf: the skip link stores the absolute triangle offset
        node.skipLink = this.triOffset + node.firstPrimitiveOffset!;
      } else {
        // interior: left child is implicitly next. traverse left, then right.
        traverse(node.leftChild!);
        traverse(node.rightChild!);
        
        node.skipLink = globalOffset + flat.length; 
      }
    };
    
    traverse(this.rootIdx);
    return flat;
  }
}