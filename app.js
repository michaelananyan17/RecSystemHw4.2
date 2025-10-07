// app.js (FIXED WITH BETTER MLP TRAINING)
class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        this.userTopRated = new Map();
        this.userFeatures = new Map();
        this.genreList = [];
        this.genreMap = new Map();
        this.model = null;
        this.mlpModel = null;
        
        this.config = {
            maxInteractions: 20000, // Reduced for faster training
            embeddingDim: 32,
            userFeatureDim: 3, // avg_rating, rating_count, rating_std - USER FEATURES
            itemFeatureDim: 19, // 19 genres in MovieLens - ITEM FEATURES (GENRES)
            batchSize: 256,
            epochs: 15,
            learningRate: 0.001,
            mlpHiddenUnits: 64 // MLP with one hidden layer
        };
        
        this.lossHistory = [];
        this.mlpLossHistory = [];
        this.isTraining = false;
        this.currentModelType = 'both'; // Default to compare both
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        // Add model type change listeners
        document.querySelectorAll('input[name="modelType"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentModelType = e.target.value;
                this.updateStatus(`Model type changed to: ${e.target.value}`);
            });
        });
        
        this.updateStatus('Click "Load Data" to start. Select "Compare Both Models" to see MLP vs Simple comparison.');
    }
    
    async loadData() {
        this.updateStatus('Loading data and extracting features...');
        
        try {
            // Load interactions
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split('\n');
            
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });
            
            // Load items with genres
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            const itemsLines = itemsText.trim().split('\n');
            
            // Parse genre list
            this.genreList = [
                "Unknown", "Action", "Adventure", "Animation", "Children's", 
                "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                "Sci-Fi", "Thriller", "War", "Western"
            ];
            
            this.genreList.forEach((genre, index) => {
                this.genreMap.set(genre, index);
            });
            
            itemsLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Parse genres (last 19 fields) - ITEM FEATURES
                const genres = parts.slice(5, 5 + 19).map(g => parseInt(g));
                if (genres.length !== 19) {
                    // Pad to 19 genres
                    while (genres.length < 19) genres.push(0);
                    if (genres.length > 19) genres.length = 19;
                }
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: genres,
                    genreVector: genres // Genre features for MLP
                });
            });
            
            // Create mappings and compute USER FEATURES
            this.createMappings();
            this.computeUserFeatures();
            this.findQualifiedUsers();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.userTopRated.size} users have 20+ ratings. Features extracted: User(3) + Item Genres(19).`);
            
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            console.error('Data loading error:', error);
        }
    }
    
    createMappings() {
        // Create user and item mappings to 0-based indices
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));
        
        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
        
        // Group interactions by user
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push(interaction);
        });
        
        // Sort each user's interactions by rating (desc) and timestamp (desc)
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
        
        this.userTopRated = userInteractions;
    }
    
    computeUserFeatures() {
        // Compute USER FEATURES: average rating, rating count, rating standard deviation
        this.userTopRated.forEach((interactions, userId) => {
            const ratings = interactions.map(i => i.rating);
            const avgRating = ratings.reduce((sum, r) => sum + r, 0) / ratings.length;
            const ratingCount = ratings.length;
            
            // Calculate standard deviation
            const squaredDiffs = ratings.map(r => Math.pow(r - avgRating, 2));
            const variance = squaredDiffs.reduce((sum, sd) => sum + sd, 0) / ratings.length;
            const ratingStd = Math.sqrt(variance);
            
            // Normalize features
            const normalizedFeatures = [
                avgRating / 5.0, // Normalize average rating to [0,1]
                Math.min(ratingCount / 100.0, 1.0), // Normalize count (cap at 100)
                ratingStd / 2.5 // Normalize std dev (assuming max ~2.5)
            ];
            
            this.userFeatures.set(userId, normalizedFeatures);
        });
        
        console.log('User features computed:', this.userFeatures.size, 'users');
    }
    
    findQualifiedUsers() {
        // Filter users with at least 20 ratings
        const qualifiedUsers = [];
        this.userTopRated.forEach((interactions, userId) => {
            if (interactions.length >= 20) {
                qualifiedUsers.push(userId);
            }
        });
        this.qualifiedUsers = qualifiedUsers;
    }
    
    async train() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.lossHistory = [];
        this.mlpLossHistory = [];
        
        this.updateStatus('Initializing models...');
        
        try {
            // Check if TwoTowerModel is available
            if (typeof TwoTowerModel === 'undefined') {
                throw new Error('TwoTowerModel class not found. Check if two-tower.js is loaded.');
            }

            // Get ITEM FEATURES (genres) for MLP
            const itemFeatures = [];
            for (let i = 0; i < this.itemMap.size; i++) {
                const originalItemId = this.reverseItemMap.get(i);
                const item = this.items.get(originalItemId);
                if (item && item.genreVector) {
                    itemFeatures.push(item.genreVector);
                } else {
                    // Fallback: zero vector if item not found
                    itemFeatures.push(Array(19).fill(0));
                }
            }
            
            console.log(`Item features: ${itemFeatures.length} items, dim: ${itemFeatures[0]?.length}`);
            
            // Initialize models based on selection
            const trainSimple = this.currentModelType === 'simple' || this.currentModelType === 'both';
            const trainMLP = this.currentModelType === 'mlp' || this.currentModelType === 'both';
            
            if (trainSimple) {
                this.updateStatus('Initializing Simple Embedding Model...');
                this.model = new TwoTowerModel(
                    this.userMap.size,
                    this.itemMap.size,
                    this.config.embeddingDim,
                    'simple'
                );
                console.log('Simple model initialized');
            }
            
            if (trainMLP) {
                this.updateStatus('Initializing MLP Deep Learning Model with user features and genre features...');
                this.mlpModel = new TwoTowerModel(
                    this.userMap.size,
                    this.itemMap.size,
                    this.config.embeddingDim,
                    'mlp',
                    {
                        userFeatureDim: this.config.userFeatureDim,
                        itemFeatureDim: this.config.itemFeatureDim,
                        hiddenUnits: this.config.mlpHiddenUnits
                    }
                );
                
                // Set USER FEATURES for MLP
                const userFeatureArray = [];
                for (let i = 0; i < this.userMap.size; i++) {
                    const originalUserId = this.reverseUserMap.get(i);
                    const features = this.userFeatures.get(originalUserId) || [0, 0, 0];
                    userFeatureArray.push(features);
                }
                
                console.log(`User features: ${userFeatureArray.length} users, dim: ${userFeatureArray[0]?.length}`);
                
                this.mlpModel.setUserFeatures(userFeatureArray);
                this.mlpModel.setItemFeatures(itemFeatures);
                console.log('MLP model initialized with user and genre features');
            }
            
            // Prepare training data
            const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
            const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
            
            // Shuffle training data for better convergence
            const shuffledIndices = this.shuffleArray([...userIndices.keys()]);
            const shuffledUsers = shuffledIndices.map(i => userIndices[i]);
            const shuffledItems = shuffledIndices.map(i => itemIndices[i]);
            
            this.updateStatus('Starting training... MLP uses user features + genre features.');
            
            // Training loop
            const numBatches = Math.ceil(shuffledUsers.length / this.config.batchSize);
            
            for (let epoch = 0; epoch < this.config.epochs; epoch++) {
                let epochLoss = 0;
                let mlpEpochLoss = 0;
                let simpleBatches = 0;
                let mlpBatches = 0;
                
                for (let batch = 0; batch < numBatches; batch++) {
                    const start = batch * this.config.batchSize;
                    const end = Math.min(start + this.config.batchSize, shuffledUsers.length);
                    
                    const batchUsers = shuffledUsers.slice(start, end);
                    const batchItems = shuffledItems.slice(start, end);
                    
                    try {
                        if (trainSimple) {
                            const loss = await this.model.trainStep(batchUsers, batchItems);
                            if (!isNaN(loss)) {
                                epochLoss += loss;
                                simpleBatches++;
                                this.lossHistory.push(loss);
                            }
                        }
                        
                        if (trainMLP) {
                            const mlpLoss = await this.mlpModel.trainStep(batchUsers, batchItems);
                            if (!isNaN(mlpLoss)) {
                                mlpEpochLoss += mlpLoss;
                                mlpBatches++;
                                this.mlpLossHistory.push(mlpLoss);
                            }
                        }
                    } catch (error) {
                        console.error(`Training error in batch ${batch}:`, error);
                        this.updateStatus(`Error in batch ${batch}: ${error.message}`);
                        continue;
                    }
                    
                    this.updateLossChart();
                    
                    if (batch % 20 === 0) {
                        let statusMsg = `Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}`;
                        if (trainSimple && this.lossHistory.length > 0) {
                            statusMsg += `, Simple: ${this.lossHistory[this.lossHistory.length-1].toFixed(4)}`;
                        }
                        if (trainMLP && this.mlpLossHistory.length > 0) {
                            statusMsg += `, MLP: ${this.mlpLossHistory[this.mlpLossHistory.length-1].toFixed(4)}`;
                        }
                        this.updateStatus(statusMsg);
                    }
                    
                    // Allow UI to update
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
                
                if (trainSimple && simpleBatches > 0) {
                    epochLoss /= simpleBatches;
                }
                if (trainMLP && mlpBatches > 0) {
                    mlpEpochLoss /= mlpBatches;
                }
                
                let epochMsg = `Epoch ${epoch + 1}/${this.config.epochs} completed.`;
                if (trainSimple) epochMsg += ` Simple: ${epochLoss.toFixed(4)}.`;
                if (trainMLP) epochMsg += ` MLP: ${mlpEpochLoss.toFixed(4)}.`;
                
                this.updateStatus(epochMsg);
                
                // Early stopping check
                if (epoch > 5) {
                    const recentSimple = this.lossHistory.slice(-10);
                    const recentMLP = this.mlpLossHistory.slice(-10);
                    
                    if (trainSimple && this.isConverging(recentSimple)) {
                        this.updateStatus('Simple model converged, stopping early.');
                        break;
                    }
                    if (trainMLP && this.isConverging(recentMLP)) {
                        this.updateStatus('MLP model converged, stopping early.');
                        break;
                    }
                }
            }
            
            this.isTraining = false;
            document.getElementById('train').disabled = false;
            document.getElementById('test').disabled = false;
            
            this.updateStatus('Training completed! Click "Test" to compare recommendations.');
            
            // Visualize embeddings
            this.visualizeEmbeddings();
            
        } catch (error) {
            this.isTraining = false;
            document.getElementById('train').disabled = false;
            this.updateStatus(`Training error: ${error.message}`);
            console.error('Training error:', error);
        }
    }
    
    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    
    isConverging(losses, threshold = 0.001) {
        if (losses.length < 5) return false;
        const recent = losses.slice(-5);
        const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
        const variance = recent.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / recent.length;
        return variance < threshold;
    }
    
    // ... rest of the methods (updateLossChart, visualizeEmbeddings, test, etc.) remain the same ...
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
