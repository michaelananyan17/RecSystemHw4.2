// two-tower.js (COMPLETE AND SYNCHRONIZED)
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, modelType = 'simple', mlpConfig = {}) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.modelType = modelType;
        this.mlpConfig = mlpConfig;
        
        console.log(`Creating TwoTowerModel: ${modelType}, users: ${numUsers}, items: ${numItems}, embDim: ${embeddingDim}`);
        
        // Initialize embedding tables with small random values
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05), 
            true, 
            'user_embeddings'
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05), 
            true, 
            'item_embeddings'
        );
        
        // MLP-specific components
        if (modelType === 'mlp') {
            this.userFeatures = null;
            this.itemFeatures = null;
            
            const userFeatureDim = mlpConfig.userFeatureDim || 3;
            const itemFeatureDim = mlpConfig.itemFeatureDim || 19; // 19 genres in MovieLens
            const hiddenUnits = mlpConfig.hiddenUnits || 64;
            
            console.log(`Building MLP towers: user_dim=${userFeatureDim}, item_dim=${itemFeatureDim}, hidden=${hiddenUnits}, output=${embeddingDim}`);
            
            // User tower MLP
            this.userTower = this.buildMLP(
                userFeatureDim,
                hiddenUnits,
                embeddingDim,
                'user_tower'
            );
            
            // Item tower MLP  
            this.itemTower = this.buildMLP(
                itemFeatureDim,
                hiddenUnits,
                embeddingDim,
                'item_tower'
            );
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
            name: `${name}_hidden`
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'linear',
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
            const lossFn = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positives
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            const lossValue = this.optimizer.minimize(lossFn, true);
            
            return lossValue ? lossValue.dataSync()[0] : 0;
        } catch (error) {
            console.error('Error in trainStep:', error);
            throw error;
        } finally {
            // Clean up tensors
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
            // Compute dot product with all item embeddings
            const scores = tf.dot(this.itemEmbeddings, userEmbedding);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
}

console.log('TwoTowerModel class loaded');
