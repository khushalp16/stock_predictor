type PredictionResult = {
  probability: number;
  prediction: 'Up' | 'Down';
};

type NormalizationParams = {
  min: number;
  max: number;
};

// Sigmoid function for logistic regression
const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

// Cost function for logistic regression
const computeCost = (X: number[][], y: number[], theta: number[], lambda: number): number => {
  const m = X.length;
  const h = X.map(x => sigmoid(x.reduce((sum, xi, j) => sum + xi * theta[j], 0)));
  const cost = y.reduce((sum, yi, i) => sum + (yi * Math.log(h[i]) + (1 - yi) * Math.log(1 - h[i])), 0) / -m;
  const regCost = lambda * theta.slice(1).reduce((sum, t) => sum + t * t, 0) / (2 * m);
  return cost + regCost;
};

// Feature scaling (Min-Max normalization)
const normalizeFeatures = (data: number[], params?: NormalizationParams): number[] => {
  // Filter out NaN values
  const validData = data.filter(x => !isNaN(x));
  if (validData.length === 0) return data.map(() => 0.5); // Return neutral value if all data is NaN

  if (params) {
    const { min, max } = params;
    if (min === max) return data.map(() => 0.5);
    return data.map(x => isNaN(x) ? 0.5 : (x - min) / (max - min));
  }
  
  const min = Math.min(...validData);
  const max = Math.max(...validData);
  if (min === max) return data.map(() => 0.5);
  return data.map(x => isNaN(x) ? 0.5 : (x - min) / (max - min));
};

// Calculate gradient for logistic regression
const calculateGradient = (
  X: number[][],
  y: number[],
  theta: number[],
  m: number,
  lambda: number
): number[] => {
  const h = X.map(x => sigmoid(x.reduce((sum, xi, j) => sum + xi * theta[j], 0)));
  return theta.map((_, j) => {
    const gradient = (1 / m) * X.reduce((sum, x, i) => {
      return sum + (h[i] - y[i]) * x[j];
    }, 0);
    // Add regularization term (except for bias term)
    return j === 0 ? gradient : gradient + (lambda / m) * theta[j];
  });
};

export const trainModel = (
  features: number[][],
  labels: number[],
  learningRate = 0.01,
  iterations = 5000,
  convergenceThreshold = 1e-6
): { theta: number[], normalizationParams: NormalizationParams[] } => {
  const m = features.length;
  const n = features[0].length;
  
  // Initialize parameters with fixed small values
  let theta = new Array(n).fill(1);
  
  // Calculate normalization parameters for each feature
  const normalizationParams: NormalizationParams[] = [];
  for (let j = 0; j < n; j++) {
    const featureValues = features.map(row => row[j]);
    normalizationParams.push({
      min: Math.min(...featureValues),
      max: Math.max(...featureValues)
    });
  }
  
  // Normalize features
  const normalizedFeatures = features.map(row => 
    row.map((val, i) => i === 0 ? 1 : normalizeFeatures([val], normalizationParams[i])[0])
  );
  
  // Add L2 regularization to prevent overfitting
  const lambda = 0.01;
  
  // Gradient descent with convergence check
  let prevCost = Infinity;
  let noImprovementCount = 0;
  
  for (let i = 0; i < iterations; i++) {
    const gradients = calculateGradient(normalizedFeatures, labels, theta, m, lambda);
    theta = theta.map((t, j) => t - learningRate * gradients[j]);
    
    // Check for convergence
    const currentCost = computeCost(normalizedFeatures, labels, theta, lambda);
    const costDiff = Math.abs(prevCost - currentCost);
    
    if (costDiff < convergenceThreshold) {
      noImprovementCount++;
      if (noImprovementCount >= 5) break; // Stop if no improvement for 5 iterations
    } else {
      noImprovementCount = 0;
    }
    
    prevCost = currentCost;
  }
  
  return { theta, normalizationParams };
};

export const predict = (
  features: number[], 
  model: { theta: number[], normalizationParams: NormalizationParams[] }
): PredictionResult => {
  // Normalize features using the stored parameters
  const normalizedFeatures = features.map((f, i) => {
    if (i === 0) return 1; // Keep bias term as 1
    
    // Handle NaN values
    if (isNaN(f)) return 0.5;
    
    // Ensure the feature value is within the training range
    const { min, max } = model.normalizationParams[i];
    const clampedValue = Math.max(min, Math.min(max, f));
    
    // Normalize the clamped value
    const normalized = normalizeFeatures([clampedValue], model.normalizationParams[i])[0];
    
    // Ensure the normalized value is between 0 and 1
    return Math.max(0, Math.min(1, normalized));
  });
  
  const z = normalizedFeatures.reduce((sum, x, i) => sum + x * model.theta[i], 0);
  const probability = Math.max(0, Math.min(1, sigmoid(z))); // Constrain between 0 and 1
  
  // Scale probability to be between 0 and 100
  const scaledProbability = Math.round(probability * 100);
  
  return {
    probability: scaledProbability,
    prediction: probability >= 0.5 ? 'Up' : 'Down'
  };
};