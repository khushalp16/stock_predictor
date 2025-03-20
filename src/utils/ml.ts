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
  m: number
): number[] => {
  const h = X.map(x => sigmoid(x.reduce((sum, xi, j) => sum + xi * theta[j], 0)));
  return theta.map((_, j) => {
    return (1 / m) * X.reduce((sum, x, i) => {
      return sum + (h[i] - y[i]) * x[j];
    }, 0);
  });
};

export const trainModel = (
  features: number[][],
  labels: number[],
  learningRate = 0.01,
  iterations = 1000
): { theta: number[], normalizationParams: NormalizationParams[] } => {
  const m = features.length;
  const n = features[0].length;
  
  // Initialize parameters
  let theta = new Array(n).fill(0);
  
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
  
  // Gradient descent
  for (let i = 0; i < iterations; i++) {
    const gradients = calculateGradient(normalizedFeatures, labels, theta, m);
    theta = theta.map((t, j) => t - learningRate * gradients[j]);
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