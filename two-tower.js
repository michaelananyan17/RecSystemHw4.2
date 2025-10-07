// two-tower.js (FIXED VARIABLE REGISTRATION)
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, modelType = 'simple', mlpConfig = {}, modelId = '') {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.modelType = modelType;
        this.mlpConfig = mlpConfig;
        this.modelId = modelId || `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        console.log(`Creating TwoTowerModel: ${modelType}, ID: ${this.modelId}, users: ${numUsers}, items: ${numItems}, embDim: ${embeddingDim}`);
        
        // Use unique variable names to avoid registration conflicts
        const userEmbeddingName = `user_embeddings_${this.modelId}`;
        const itemEmbeddingName = `item_embeddings_${this.modelId}`;
        
        // Check if variables already exist and dispose them first
        try {
            const existingUserVars = tf.engine().state.registeredVariables;
            if (existingUserVars[userEmbeddingName]) {
                console.log(`Disposing existing variable: ${userEmbeddingName}`);
                tf.disposeVariables(userEmbeddingName);
            }
            if (existingUserVars[itemEmbeddingName]) {
                console.log(`Disposing existing variable: ${itemEmbeddingName}`);
                tf.disposeVariables(itemEmbeddingName);
            }
        } catch (e) {
            console.log('No existing variables to dispose');
        }
        
        // Initialize embedding tables with small random values
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.01),
            true, 
            userEmbeddingName
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.01),
            true, 
            itemEmbeddingName
        );
        
        // MLP-specific components
        if (modelType === 'mlp') {
            this.userFeatures = null;
            this.itemFeatures = null;
            
            const userFeatureDim = mlpConfig.userFeatureDim || 3;
            const itemFeatureDim = mlpConfig.itemFeatureDim || 19;
            const hiddenUnits = mlpConfig.hiddenUnits || 64;
            
            console.log(`Building MLP towers: user_dim=${userFeatureDim}, item_dim=${itemFeatureDim}, hidden=${hiddenUnits}, output=${embeddingDim}`);
            
            // User tower MLP
            this.userTower = this.buildMLP(
                userFeatureDim,
                hiddenUnits,
                embeddingDim,
                `user_tower_${this.modelId}`
            );
            
            // Item tower MLP  
            this.itemTower = this.buildMLP(
                itemFeatureDim,
                hiddenUnits,
                embeddingDim,
                `item_tower_${this.modelId}`
            );
            
            // Separate optimizer for MLP weights
            this.mlpOptimizer = tf.train.adam(0.0005);
        }
        
        // Adam optimizer for stable training
        this.optimizer = tf.train.adam(0.001);
        
        console.log('TwoTowerModel created successfully');
    }
    
    buildMLP(inputDim, hiddenUnits, outputDim, name) {
        console.log(`Building MLP ${name}: input=${inputDim}, hidden=${hiddenUnits}, output=${outputDim}`);
        
        const model = tf.sequential();
        
        // Hidden layer
        model.add(tf.layers.dense({
            units: hiddenUnits,
            activation: 'relu',
            inputShape: [inputDim],
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros',
            name: `${name}_hidden`
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros',
            name: `${name}_output`
        }));
        
        console.log(`MLP ${name} built successfully`);
        return model;
    }
    
    setUserFeatures(userFeatures) {
        if (this.modelType === 'mlp') {
            console.log(`Setting user features: ${userFeatures.length} users, feature dim: ${userFeatures[0]?.length}`);
            this.userFeatures = tf.tensor2d(userFeatures);
        }
    }
    
    setItemFeatures(itemFeatures) {
        if (this.modelType === 'mlp') {
            console.log(`Setting item features: ${itemFeatures.length} items, feature dim: ${itemFeatures[0]?.length}`);
            this.itemFeatures = tf.tensor2d(itemFeatures);
        }
    }
    
    // User tower: simple embedding lookup or MLP
    userForward(userIndices) {
        if (this.modelType === 'simple') {
            return tf.gather(this.userEmbeddings, userIndices);
        } else {
            if (!this.userFeatures) {
                throw new Error('User features not set for MLP model');
            }
            const features = tf.gather(this.userFeatures, userIndices);
            return this.userTower.apply(features);
        }
    }
    
    // Item tower: simple embedding lookup or MLP
    itemForward(itemIndices) {
        if (this.modelType === 'simple') {
            return tf.gather(this.itemEmbeddings, itemIndices);
        } else {
            if (!this.itemFeatures) {
                throw new Error('Item features not set for MLP model');
            }
            const features = tf.gather(this.itemFeatures, itemIndices);
            return this.itemTower.apply(features);
        }
    }
    
    // Scoring function: dot product between user and item embeddings
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        const userTensor = tf.tensor1d(userIndices, 'int32');
        const itemTensor = tf.tensor1d(itemIndices, 'int32');
        
        try {
            if (this.modelType === 'simple') {
                const lossFn = () => {
                    const userEmbs = this.userForward(userTensor);
                    const itemEmbs = this.itemForward(itemTensor);
                    
                    const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                    
                    const labels = tf.oneHot(
                        tf.range(0, userIndices.length, 1, 'int32'), 
                        userIndices.length
                    );
                    
                    const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                    return loss;
                };
                
                const lossValue = this.optimizer.minimize(lossFn, true);
                return lossValue ? lossValue.dataSync()[0] : 0;
                
            } else {
                const lossFn = () => {
                    const userEmbs = this.userForward(userTensor);
                    const itemEmbs = this.itemForward(itemTensor);
                    
                    const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                    
                    const labels = tf.oneHot(
                        tf.range(0, userIndices.length, 1, 'int32'), 
                        userIndices.length
                    );
                    
                    const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                    return loss;
                };
                
                const lossValue = this.mlpOptimizer.minimize(lossFn, true);
                return lossValue ? lossValue.dataSync()[0] : 0;
            }
        } catch (error) {
            console.error('Error in trainStep:', error);
            throw error;
        } finally {
            userTensor.dispose();
            itemTensor.dispose();
        }
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            if (this.modelType === 'simple') {
                const scores = tf.dot(this.itemEmbeddings, userEmbedding);
                return scores.dataSync();
            } else {
                const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
                const itemTensor = tf.tensor1d(allItemIndices, 'int32');
                try {
                    const itemEmbs = this.itemForward(itemTensor);
                    const scores = tf.dot(itemEmbs, userEmbedding);
                    return scores.dataSync();
                } finally {
                    itemTensor.dispose();
                }
            }
        });
    }
    
    getItemEmbeddings() {
        if (this.modelType === 'simple') {
            return this.itemEmbeddings;
        } else {
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const itemTensor = tf.tensor1d(allItemIndices, 'int32');
            const embeddings = this.itemForward(itemTensor);
            itemTensor.dispose();
            return embeddings;
        }
    }
    
    // Clean up method to dispose variables
    dispose() {
        console.log(`Disposing model: ${this.modelId}`);
        try {
            tf.dispose(this.userEmbeddings);
            tf.dispose(this.itemEmbeddings);
            if (this.userFeatures) tf.dispose(this.userFeatures);
            if (this.itemFeatures) tf.dispose(this.itemFeatures);
            if (this.userTower) this.userTower.dispose();
            if (this.itemTower) this.itemTower.dispose();
        } catch (e) {
            console.warn('Error disposing model:', e);
        }
    }
}

console.log('TwoTowerModel class loaded');
