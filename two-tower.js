// Simple Two Tower Embedding Model
class TwoTowerModel {
    constructor(embeddingDim = 8) {
        this.embeddingDim = embeddingDim;
        this.userEmbeddings = new Map();
        this.movieEmbeddings = new Map();
        this.userBiases = new Map();
        this.movieBiases = new Map();
        this.globalBias = 3.0;
        this.learningRate = 0.01;
        this.regularization = 0.001;
        this.isTrained = false;
    }

    initializeParameters(userIds, movieIds) {
        userIds.forEach(userId => {
            if (!this.userEmbeddings.has(userId)) {
                this.userEmbeddings.set(userId, this.randomArray(this.embeddingDim, 0.1));
                this.userBiases.set(userId, 0);
            }
        });

        movieIds.forEach(movieId => {
            if (!this.movieEmbeddings.has(movieId)) {
                this.movieEmbeddings.set(movieId, this.randomArray(this.embeddingDim, 0.1));
                this.movieBiases.set(movieId, 0);
            }
        });
    }

    randomArray(length, scale = 1.0) {
        return Array.from({length}, () => (Math.random() - 0.5) * 2 * scale);
    }

    dotProduct(vec1, vec2) {
        return vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
    }

    predict(userId, movieId) {
        const userEmbedding = this.userEmbeddings.get(userId);
        const movieEmbedding = this.movieEmbeddings.get(movieId);
        const userBias = this.userBiases.get(userId) || 0;
        const movieBias = this.movieBiases.get(movieId) || 0;

        if (!userEmbedding || !movieEmbedding) {
            return this.globalBias;
        }

        const interaction = this.dotProduct(userEmbedding, movieEmbedding);
        const prediction = this.globalBias + userBias + movieBias + interaction;
        
        // Constrain to rating range
        return Math.max(1, Math.min(5, prediction));
    }

    async train(interactions, epochs = 100) {
        console.log('Training Simple Embedding Model...');
        
        const userIds = [...new Set(interactions.map(i => i.userId))];
        const movieIds = [...new Set(interactions.map(i => i.movieId))];
        
        this.initializeParameters(userIds, movieIds);
        
        const losses = [];
        let previousLoss = Infinity;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let count = 0;

            // Shuffle interactions
            const shuffled = [...interactions].sort(() => Math.random() - 0.5);
            
            for (const {userId, movieId, rating} of shuffled) {
                const prediction = this.predict(userId, movieId);
                const error = prediction - rating;
                
                this.updateParameters(userId, movieId, error);
                
                totalLoss += error * error;
                count++;
            }

            const avgLoss = totalLoss / count;
            losses.push(avgLoss);
            
            // Early stopping if improvement is minimal
            if (Math.abs(previousLoss - avgLoss) < 0.0001 && epoch > 10) {
                console.log(`Early stopping at epoch ${epoch}`);
                break;
            }
            previousLoss = avgLoss;
            
            if (epoch % 20 === 0) {
                console.log(`Simple Model Epoch ${epoch}, Loss: ${avgLoss.toFixed(4)}`);
            }
        }

        this.isTrained = true;
        console.log('Simple Embedding Model training completed');
        return losses;
    }

    updateParameters(userId, movieId, error) {
        const userEmbedding = this.userEmbeddings.get(userId);
        const movieEmbedding = this.movieEmbeddings.get(movieId);
        
        if (!userEmbedding || !movieEmbedding) return;

        // Update embeddings with regularization
        for (let i = 0; i < this.embeddingDim; i++) {
            const userGrad = error * movieEmbedding[i] + this.regularization * userEmbedding[i];
            const movieGrad = error * userEmbedding[i] + this.regularization * movieEmbedding[i];
            
            userEmbedding[i] -= this.learningRate * userGrad;
            movieEmbedding[i] -= this.learningRate * movieGrad;
        }

        // Update biases
        const userBias = this.userBiases.get(userId) || 0;
        const movieBias = this.movieBiases.get(movieId) || 0;
        
        this.userBiases.set(userId, userBias - this.learningRate * error);
        this.movieBiases.set(movieId, movieBias - this.learningRate * error);
    }

    async recommend(userId, allMovieIds, topK = 5) {
        if (!this.isTrained) {
            throw new Error('Simple model not trained yet');
        }

        const scores = allMovieIds.map(movieId => ({
            movieId,
            score: this.predict(userId, movieId)
        }));

        return scores.sort((a, b) => b.score - a.score).slice(0, topK);
    }

    getMovieEmbeddings() {
        return Array.from(this.movieEmbeddings.entries()).map(([movieId, vector]) => ({
            movieId,
            vector
        }));
    }
}

// Advanced MLP Deep Learning Model
class MLPModel {
    constructor(inputDim = 64, hiddenLayers = [128, 64, 32], outputDim = 1) {
        this.inputDim = inputDim;
        this.hiddenLayers = hiddenLayers;
        this.outputDim = outputDim;
        this.weights = [];
        this.biases = [];
        this.learningRate = 0.001;
        this.regularization = 0.0001;
        this.dropoutRate = 0.2;
        this.isTrained = false;
        
        this.initializeNetwork();
    }

    initializeNetwork() {
        const dimensions = [this.inputDim, ...this.hiddenLayers, this.outputDim];
        this.weights = [];
        this.biases = [];

        for (let i = 0; i < dimensions.length - 1; i++) {
            const inputSize = dimensions[i];
            const outputSize = dimensions[i + 1];
            
            // He initialization for ReLU
            const scale = Math.sqrt(2.0 / inputSize);
            this.weights.push(
                Array.from({length: outputSize}, () => 
                    Array.from({length: inputSize}, () => (Math.random() - 0.5) * 2 * scale)
                )
            );
            this.biases.push(Array.from({length: outputSize}, () => 0));
        }
    }

    encodeInput(userId, movieId) {
        // Create a rich feature encoding
        const encoding = new Array(this.inputDim).fill(0);
        
        // User encoding (first 32 dimensions)
        const userHash = this.stringToHash(userId);
        for (let i = 0; i < 16; i++) {
            encoding[(userHash + i) % 32] = 1;
        }
        
        // Movie encoding (next 32 dimensions)
        const movieHash = this.stringToHash(movieId);
        for (let i = 0; i < 16; i++) {
            encoding[32 + (movieHash + i) % 32] = 1;
        }
        
        return encoding;
    }

    stringToHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }

    relu(x) {
        return Math.max(0, x);
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0.01; // Leaky ReLU derivative
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    forward(input, training = false) {
        let current = input;
        const activations = [input];
        const preActivations = [];
        const dropoutMasks = [];

        // Hidden layers with dropout
        for (let i = 0; i < this.weights.length - 1; i++) {
            const layerPreActivation = this.matrixVectorMultiply(this.weights[i], current);
            this.vectorAdd(layerPreActivation, this.biases[i]);
            preActivations.push([...layerPreActivation]);
            
            // Apply ReLU activation
            current = layerPreActivation.map(this.relu);
            
            // Apply dropout during training
            if (training) {
                const mask = current.map(() => Math.random() > this.dropoutRate ? 1 : 0);
                current = current.map((val, idx) => val * mask[idx]);
                dropoutMasks.push(mask);
            }
            
            activations.push(current);
        }

        // Output layer (linear activation for regression)
        const outputPreActivation = this.matrixVectorMultiply(this.weights[this.weights.length - 1], current);
        this.vectorAdd(outputPreActivation, this.biases[this.biases.length - 1]);
        preActivations.push([...outputPreActivation]);
        
        const output = outputPreActivation[0]; // Single output
        activations.push([output]);

        return { 
            output, 
            activations, 
            preActivations, 
            dropoutMasks 
        };
    }

    matrixVectorMultiply(matrix, vector) {
        return matrix.map(row => 
            row.reduce((sum, weight, j) => sum + weight * vector[j], 0)
        );
    }

    vectorAdd(vector, other) {
        for (let i = 0; i < vector.length; i++) {
            vector[i] += other[i];
        }
    }

    predict(userId, movieId) {
        const input = this.encodeInput(userId, movieId);
        const { output } = this.forward(input, false);
        
        // Properly invert the log scaling used in training
        const scaledOutput = 1 + 4 * this.sigmoid(output);
        return Math.max(1, Math.min(5, scaledOutput));
    }

    async train(interactions, epochs = 100) {
        console.log('Training MLP Deep Learning Model...');
        
        const losses = [];
        let previousLoss = Infinity;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let count = 0;

            // Shuffle with different random seed each epoch
            const shuffled = [...interactions].sort(() => Math.random() - 0.5);
            
            for (const {userId, movieId, rating} of shuffled) {
                const input = this.encodeInput(userId, movieId);
                const { output, activations, preActivations, dropoutMasks } = this.forward(input, true);
                
                // Scale target to match output range using logit transformation
                const scaledTarget = Math.log((rating - 1) / (5 - rating));
                const error = output - scaledTarget;
                
                totalLoss += error * error;
                count++;
                
                // Backpropagation
                this.backward(input, error, activations, preActivations, dropoutMasks);
            }

            const avgLoss = totalLoss / count;
            losses.push(avgLoss);
            
            // Adaptive learning rate and early stopping
            if (avgLoss > previousLoss) {
                this.learningRate *= 0.95;
            }
            
            if (Math.abs(previousLoss - avgLoss) < 0.0001 && epoch > 20) {
                console.log(`MLP Early stopping at epoch ${epoch}`);
                break;
            }
            previousLoss = avgLoss;
            
            if (epoch % 20 === 0) {
                console.log(`MLP Epoch ${epoch}, Loss: ${avgLoss.toFixed(4)}, LR: ${this.learningRate.toFixed(6)}`);
            }
        }

        this.isTrained = true;
        console.log('MLP Deep Learning Model training completed');
        return losses;
    }

    backward(input, error, activations, preActivations, dropoutMasks) {
        let delta = [error];
        const weightGradients = [];
        const biasGradients = [];

        // Backward pass through layers
        for (let i = this.weights.length - 1; i >= 0; i--) {
            const activation = activations[i];
            const preActivation = preActivations[i];
            
            // Calculate gradients for this layer
            const layerWeightGradients = delta.map(d => 
                activation.map(a => d * a)
            );
            
            const layerBiasGradients = [...delta];
            
            weightGradients.unshift(layerWeightGradients);
            biasGradients.unshift(layerBiasGradients);
            
            // Propagate delta to previous layer (if not input layer)
            if (i > 0) {
                const newDelta = new Array(activation.length).fill(0);
                for (let j = 0; j < activation.length; j++) {
                    for (let k = 0; k < delta.length; k++) {
                        newDelta[j] += delta[k] * this.weights[i][k][j] * this.reluDerivative(preActivation[j]);
                    }
                    // Apply dropout mask
                    if (dropoutMasks[i - 1]) {
                        newDelta[j] *= dropoutMasks[i - 1][j];
                    }
                }
                delta = newDelta;
            }
        }

        // Update weights and biases with regularization
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    const grad = weightGradients[i][j][k] + this.regularization * this.weights[i][j][k];
                    this.weights[i][j][k] -= this.learningRate * grad;
                }
                this.biases[i][j] -= this.learningRate * biasGradients[i][j];
            }
        }
    }

    async recommend(userId, allMovieIds, topK = 5) {
        if (!this.isTrained) {
            throw new Error('MLP model not trained yet');
        }

        const scores = allMovieIds.map(movieId => ({
            movieId,
            score: this.predict(userId, movieId)
        }));

        return scores.sort((a, b) => b.score - a.score).slice(0, topK);
    }
}

// Export for CommonJS
module.exports = { TwoTowerModel, MLPModel };
