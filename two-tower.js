// app.js (FIXED - with better error handling)
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
            maxInteractions: 80000,
            embeddingDim: 32,
            userFeatureDim: 3, // avg_rating, rating_count, rating_std
            itemFeatureDim: 19, // FIXED: 19 genres in MovieLens (including Unknown)
            batchSize: 256, // Reduced for stability
            epochs: 15, // Reduced for testing
            learningRate: 0.001,
            mlpHiddenUnits: 32 // Reduced for stability
        };
        
        this.lossHistory = [];
        this.mlpLossHistory = [];
        this.isTraining = false;
        this.currentModelType = 'simple';
        
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
        
        this.updateStatus('Click "Load Data" to start');
    }
    
    async loadData() {
        this.updateStatus('Loading data...');
        
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
            
            // Parse genre list from u.genre file structure (known MovieLens genres)
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
                
                // Parse genres (last 19 fields) - FIXED: ensure we get exactly 19 genres
                const genres = parts.slice(5, 5 + 19).map(g => parseInt(g));
                if (genres.length !== 19) {
                    console.warn(`Item ${itemId} has ${genres.length} genres, padding to 19`);
                    // Pad or truncate to 19 genres
                    while (genres.length < 19) genres.push(0);
                    if (genres.length > 19) genres.length = 19;
                }
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: genres,
                    genreVector: genres // Use as item features
                });
            });
            
            // Create mappings and find users with sufficient ratings
            this.createMappings();
            this.computeUserFeatures();
            this.findQualifiedUsers();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.userTopRated.size} users have 20+ ratings.`);
            
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
        
        // Group interactions by user and find top rated movies
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
        // Compute user features: average rating, rating count, rating standard deviation
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
            // Get item feature matrix - FIXED: ensure proper dimensions
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
                this.updateStatus('Initializing simple model...');
                this.model = new TwoTowerModel(
                    this.userMap.size,
                    this.itemMap.size,
                    this.config.embeddingDim,
                    'simple'
                );
            }
            
            if (trainMLP) {
                this.updateStatus('Initializing MLP model...');
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
                
                // Set feature matrices
                const userFeatureArray = [];
                for (let i = 0; i < this.userMap.size; i++) {
                    const originalUserId = this.reverseUserMap.get(i);
                    const features = this.userFeatures.get(originalUserId) || [0, 0, 0];
                    userFeatureArray.push(features);
                }
                
                console.log(`User features: ${userFeatureArray.length} users, dim: ${userFeatureArray[0]?.length}`);
                
                this.mlpModel.setUserFeatures(userFeatureArray);
                this.mlpModel.setItemFeatures(itemFeatures);
            }
            
            // Prepare training data
            const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
            const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
            
            this.updateStatus('Starting training...');
            
            // Training loop
            const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
            
            for (let epoch = 0; epoch < this.config.epochs; epoch++) {
                let epochLoss = 0;
                let mlpEpochLoss = 0;
                let simpleBatches = 0;
                let mlpBatches = 0;
                
                for (let batch = 0; batch < numBatches; batch++) {
                    const start = batch * this.config.batchSize;
                    const end = Math.min(start + this.config.batchSize, userIndices.length);
                    
                    const batchUsers = userIndices.slice(start, end);
                    const batchItems = itemIndices.slice(start, end);
                    
                    try {
                        if (trainSimple) {
                            const loss = await this.model.trainStep(batchUsers, batchItems);
                            epochLoss += loss;
                            simpleBatches++;
                            this.lossHistory.push(loss);
                        }
                        
                        if (trainMLP) {
                            const mlpLoss = await this.mlpModel.trainStep(batchUsers, batchItems);
                            mlpEpochLoss += mlpLoss;
                            mlpBatches++;
                            this.mlpLossHistory.push(mlpLoss);
                        }
                    } catch (error) {
                        console.error(`Training error in batch ${batch}:`, error);
                        this.updateStatus(`Error in batch ${batch}: ${error.message}`);
                        // Continue with next batch
                        continue;
                    }
                    
                    this.updateLossChart();
                    
                    if (batch % 10 === 0) {
                        let statusMsg = `Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}`;
                        if (trainSimple && this.lossHistory.length > 0) statusMsg += `, Simple Loss: ${this.lossHistory[this.lossHistory.length-1].toFixed(4)}`;
                        if (trainMLP && this.mlpLossHistory.length > 0) statusMsg += `, MLP Loss: ${this.mlpLossHistory[this.mlpLossHistory.length-1].toFixed(4)}`;
                        this.updateStatus(statusMsg);
                    }
                    
                    // Allow UI to update
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
                
                if (trainSimple && simpleBatches > 0) {
                    epochLoss /= simpleBatches;
                }
                if (trainMLP && mlpBatches > 0) {
                    mlpEpochLoss /= mlpBatches;
                }
                
                let epochMsg = `Epoch ${epoch + 1}/${this.config.epochs} completed.`;
                if (trainSimple) epochMsg += ` Simple avg loss: ${epochLoss.toFixed(4)}.`;
                if (trainMLP) epochMsg += ` MLP avg loss: ${mlpEpochLoss.toFixed(4)}.`;
                
                this.updateStatus(epochMsg);
            }
            
            this.isTraining = false;
            document.getElementById('train').disabled = false;
            document.getElementById('test').disabled = false;
            
            this.updateStatus('Training completed! Click "Test" to see recommendations.');
            
            // Visualize embeddings from the main model
            this.visualizeEmbeddings();
            
        } catch (error) {
            this.isTraining = false;
            document.getElementById('train').disabled = false;
            this.updateStatus(`Training error: ${error.message}`);
            console.error('Training error:', error);
        }
    }
    
    // ... rest of the methods remain the same (updateLossChart, visualizeEmbeddings, test, etc.) ...
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log('Status:', message);
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
