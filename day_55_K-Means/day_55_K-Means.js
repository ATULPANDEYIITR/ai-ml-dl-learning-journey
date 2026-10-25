'use strict';

/**
 * K-Means clustering using K-Means++ initialization, multiple restarts,
 * centroid updates, inertia, prediction, and elbow-based K inspection.
 */

const Initialization = Object.freeze({
  RANDOM: 'random',
  KMEANS_PLUS_PLUS: 'kmeans++',
});

function validatePoints(input) {
  if (!Array.isArray(input) || input.length === 0) {
    throw new TypeError('At least one data point is required.');
  }

  if (!Array.isArray(input[0]) || input[0].length === 0) {
    throw new TypeError('Points must contain at least one feature.');
  }

  const dimension = input[0].length;

  return input.map((point, index) => {
    if (!Array.isArray(point) || point.length !== dimension) {
      throw new TypeError(`Point ${index} has inconsistent dimensions.`);
    }

    return point.map((value) => {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new TypeError('Every feature must be a finite number.');
      }
      return value;
    });
  });
}

function squaredDistance(a, b) {
  return a.reduce((total, value, i) => total + (value - b[i]) ** 2, 0);
}

// Mulberry32 provides reproducible pseudo-random values for experiments.
// It is not a cryptographic random-number generator.
function seededRandom(seed) {
  let state = seed >>> 0;

  return function random() {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function pickWeighted(weights, random) {
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (total === 0) return -1;

  let threshold = random() * total;

  for (let i = 0; i < weights.length; i += 1) {
    threshold -= weights[i];
    if (threshold <= 0) return i;
  }

  return weights.length - 1;
}

class KMeans {
  constructor({
    k,
    initialization = Initialization.KMEANS_PLUS_PLUS,
    maxIterations = 300,
    tolerance = 1e-6,
    seed = 42,
    restarts = 5,
  }) {
    if (!Number.isInteger(k) || k < 1) {
      throw new RangeError('K must be a positive integer.');
    }
    if (!Number.isInteger(maxIterations) || maxIterations < 1) {
      throw new RangeError('maxIterations must be positive.');
    }
    if (!Number.isInteger(restarts) || restarts < 1) {
      throw new RangeError('restarts must be positive.');
    }
    if (!Number.isFinite(tolerance) || tolerance < 0) {
      throw new RangeError('tolerance must be finite and non-negative.');
    }
    if (!Object.values(Initialization).includes(initialization)) {
      throw new TypeError('Unknown initialization strategy.');
    }

    this.k = k;
    this.initialization = initialization;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
    this.seed = seed;
    this.restarts = restarts;
    this.result = null;
  }

  initialize(points, random) {
    if (this.k > points.length) {
      throw new RangeError('K cannot exceed the number of data points.');
    }

    if (this.initialization === Initialization.RANDOM) {
      const indices = points.map((_, i) => i);

      for (let i = indices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      return indices.slice(0, this.k).map((i) => [...points[i]]);
    }

    const centers = [[...points[Math.floor(random() * points.length)]]];

    while (centers.length < this.k) {
      const weights = points.map((point) =>
        Math.min(...centers.map((center) => squaredDistance(point, center)))
      );

      const selectedIndex = pickWeighted(weights, random);

      if (selectedIndex < 0) {
        const unused = points.filter(
          (point) => !centers.some(
            (center) => center.every((v, i) => v === point[i])
          )
        );
        centers.push([...(unused.length
          ? unused[Math.floor(random() * unused.length)]
          : points[Math.floor(random() * points.length)])]);
      } else {
        centers.push([...points[selectedIndex]]);
      }
    }

    return centers;
  }

  fitOnce(points, random) {
    let centers = this.initialize(points, random);
    let labels = new Array(points.length).fill(-1);
    let converged = false;
    let iterations = 0;

    for (iterations = 1; iterations <= this.maxIterations; iterations += 1) {
      const nextLabels = points.map((point) => {
        let nearest = 0;
        let distance = squaredDistance(point, centers[0]);

        for (let c = 1; c < this.k; c += 1) {
          const candidate = squaredDistance(point, centers[c]);
          if (candidate < distance) {
            distance = candidate;
            nearest = c;
          }
        }

        return nearest;
      });

      const sums = Array.from(
        { length: this.k },
        () => new Array(points[0].length).fill(0)
      );
      const counts = new Array(this.k).fill(0);

      points.forEach((point, i) => {
        const cluster = nextLabels[i];
        counts[cluster] += 1;
        point.forEach((value, feature) => {
          sums[cluster][feature] += value;
        });
      });

      const nextCenters = sums.map((sum, cluster) => {
        if (counts[cluster] === 0) {
          // Reseed an empty cluster with a point far from all current centers.
          let farthest = points[0];
          let farthestDistance = -1;

          points.forEach((point) => {
            const nearestDistance = Math.min(
              ...centers.map((center) => squaredDistance(point, center))
            );
            if (nearestDistance > farthestDistance) {
              farthest = point;
              farthestDistance = nearestDistance;
            }
          });

          return [...farthest];
        }

        return sum.map((value) => value / counts[cluster]);
      });

      let maxShift = 0;
      nextCenters.forEach((center, i) => {
        maxShift = Math.max(maxShift, squaredDistance(center, centers[i]));
      });

      const stable = nextLabels.every((label, i) => label === labels[i]);
      centers = nextCenters;
      labels = nextLabels;

      if (stable || maxShift <= this.tolerance ** 2) {
        converged = true;
        break;
      }
    }

    // Reassign using the final centers to keep labels consistent.
    labels = points.map((point) => {
      let nearest = 0;
      for (let c = 1; c < this.k; c += 1) {
        if (
          squaredDistance(point, centers[c]) <
          squaredDistance(point, centers[nearest])
        ) {
          nearest = c;
        }
      }
      return nearest;
    });

    const inertia = points.reduce(
      (sum, point, i) => sum + squaredDistance(point, centers[labels[i]]),
      0
    );

    return {
      centroids: centers,
      labels,
      inertia,
      iterations: Math.min(iterations, this.maxIterations),
      converged,
    };
  }

  fit(input) {
    const points = validatePoints(input);

    if (this.k > points.length) {
      throw new RangeError('K cannot exceed the number of data points.');
    }

    const masterRandom = seededRandom(this.seed);
    let best = null;

    for (let attempt = 0; attempt < this.restarts; attempt += 1) {
      const runSeed = Math.floor(masterRandom() * 4294967296);
      const candidate = this.fitOnce(points, seededRandom(runSeed));

      if (!best || candidate.inertia < best.inertia) best = candidate;
    }

    this.result = best;
    return {
      ...best,
      centroids: best.centroids.map((center) => [...center]),
      labels: [...best.labels],
    };
  }

  predict(input) {
    if (!this.result) throw new Error('Call fit before predict.');

    const points = validatePoints(input);
    const dimension = this.result.centroids[0].length;

    if (points.some((point) => point.length !== dimension)) {
      throw new RangeError('Prediction dimensions differ from training data.');
    }

    return points.map((point) => {
      let nearest = 0;
      for (let c = 1; c < this.k; c += 1) {
        if (
          squaredDistance(point, this.result.centroids[c]) <
          squaredDistance(point, this.result.centroids[nearest])
        ) {
          nearest = c;
        }
      }
      return nearest;
    });
  }
}

async function evaluateCandidates(points, candidates, seed = 42) {
  // Yield between candidates so a calling service can remain responsive.
  const results = [];

  for (const k of candidates) {
    const model = new KMeans({ k, seed, restarts: 5 });
    const result = model.fit(points);
    results.push({ k, inertia: result.inertia });
    await new Promise((resolve) => setImmediate(resolve));
  }

  return results;
}

async function main() {
  const points = [
    [1.0, 1.2], [1.3, 0.8], [0.7, 1.1], [1.2, 1.5],
    [5.0, 5.2], [5.4, 4.8], [4.7, 5.1], [5.2, 5.5],
    [9.0, 1.0], [9.4, 1.4], [8.6, 0.7], [9.2, 0.6],
  ];

  const model = new KMeans({ k: 3, seed: 42, restarts: 10 });
  const result = model.fit(points);

  console.log('Centroids:', result.centroids);
  console.log('Cluster assignments:', result.labels);
  console.log('Inertia:', result.inertia.toFixed(4));
  console.log('Converged:', result.converged);

  console.log(
    'Elbow candidates:',
    await evaluateCandidates(points, [1, 2, 3, 4, 5, 6])
  );

  if (result.labels.length !== points.length || !Number.isFinite(result.inertia)) {
    throw new Error('Clustering invariants failed.');
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Clustering failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { KMeans, Initialization, validatePoints, squaredDistance };
