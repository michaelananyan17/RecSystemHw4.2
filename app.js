// app.js (COMPLETE FIXED VERSION)
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
        this.modelCounter = 0;
        
        this.config = {
            maxInteractions: 20000,
            embeddingDim: 32,
            userFeatureDim: 3,
            itemFeatureDim: 19,
            batchSize: 256,
            epochs: 15,
            learningRate: 0.001,
            mlpHiddenUnits: 64
        };
        
        this.lossHistory = [];
        this.mlpLossHistory = [];
        this.isTraining = false;
        this.currentModelType = 'both';
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
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
            // Clear any existing models and variables
            this.disposeExistingModels();
            
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
                
                // Parse genres
                const genres = parts.slice(5, 5 + 19).map(g => parseInt(g));
                if (genres.length !== 19) {
                    while (genres.length < 19) genres.push(0);
                    if (genres.length > 19) genres.length = 19;
                }
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: genres,
                    genreVector: genres
                });
            });
            
            // Create mappings and compute features
            this.createMappings();
            this.computeUserFeatures();
            this.findQualifiedUsers();
            
            this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.userTopRated.size} users have 20+ ratings. Features: User(3) + Genres(19)`);
            
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
            console.error('Data loading error:', error);
        }
    }
    
    disposeExistingModels() {
        this.updateStatus('Cleaning up existing models...');
        try {
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            if (this.mlpModel) {
                this.mlpModel.dispose();
                this.mlpModel = null;
            }
            // Clear TensorFlow.js engine state
            tf.engine().startScope();
            tf.engine().endScope();
            console.log('Existing models disposed');
        } catch (e) {
            console.warn('Error disposing models:', e);
        }
    }
    
    createMappings() {
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
        
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push(interaction);
        });
        
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
        
        this.userTopRated = userInteractions;
    }
    
    computeUserFeatures() {
        this.userTopRated.forEach((interactions, userId) => {
            const ratings = interactions.map(i => i.rating);
            const avgRating = ratings.reduce((sum, r) => sum + r, 0) / ratings.length;
            const ratingCount = ratings.length;
            
            const squaredDiffs = ratings.map(r => Math.pow(r - avgRating, 2));
            const variance = squaredDiffs.reduce((sum, sd) => sum + sd, 0) / ratings.length;
            const ratingStd = Math.sqrt(variance);
            
            const normalizedFeatures = [
                avgRating / 5.0,
                Math.min(ratingCount / 100.0, 1.0),
                ratingStd / 2.5
            ];
            
            this.userFeatures.set(userId, normalizedFeatures);
        });
        
        console.log('User features computed:', this.userFeatures.size, 'users');
    }
    
    findQualifiedUsers() {
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
        
        // Dispose any existing models before creating new ones
        this.disposeExistingModels();
        
        this.updateStatus('Initializing models with unique variable names...');
        
        try {
            // Get item features for MLP
            const itemFeatures = [];
            for (let i = 0; i < this.itemMap.size; i++) {
                const originalItemId = this.reverseItemMap.get(i);
                const item = this.items.get(originalItemId);
                if (item && item.genreVector) {
                    itemFeatures.push(item.genreVector);
                } else {
                    itemFeatures.push(Array(19).fill(0));
                }
            }
            
            const trainSimple = this.currentModelType === 'simple' || this.currentModelType === 'both';
            const trainMLP = this.currentModelType === 'mlp' || this.currentModelType === 'both';
            
            if (trainSimple) {
                this.updateStatus('Initializing Simple Embedding Model...');
                this.model = new TwoTowerModel(
                    this.userMap.size,
                    this.itemMap.size,
                    this.config.embeddingDim,
                    'simple',
                    {},
                    `simple_${this.modelCounter++}`
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
                    },
                    `mlp_${this.modelCounter++}`
                );
                
                // Set user features for MLP
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
            
            // Shuffle data for better training
            const shuffledIndices = this.shuffleArray([...userIndices.keys()]);
            const shuffledUsers = shuffledIndices.map(i => userIndices[i]);
            const shuffledItems = shuffledIndices.map(i => itemIndices[i]);
            
            this.updateStatus('Starting training... MLP uses user behavior features + movie genre features.');
            
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
                        if (trainSimple && this.model) {
                            const loss = await this.model.trainStep(batchUsers, batchItems);
                            if (!isNaN(loss)) {
                                epochLoss += loss;
                                simpleBatches++;
                                this.lossHistory.push(loss);
                            }
                        }
                        
                        if (trainMLP && this.mlpModel) {
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
                        if (!trainMLP) break;
                    }
                    if (trainMLP && this.isConverging(recentMLP)) {
                        this.updateStatus('MLP model converged, stopping early.');
                        if (!trainSimple) break;
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
    
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw simple model loss
        if (this.lossHistory.length > 0) {
            const maxLoss = Math.max(...this.lossHistory);
            const minLoss = Math.min(...this.lossHistory);
            const range = maxLoss - minLoss || 1;
            
            ctx.strokeStyle = '#007acc';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            this.lossHistory.forEach((loss, index) => {
                const x = (index / this.lossHistory.length) * canvas.width;
                const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        }
        
        // Draw MLP model loss
        if (this.mlpLossHistory.length > 0) {
            const maxLoss = Math.max(...this.mlpLossHistory);
            const minLoss = Math.min(...this.mlpLossHistory);
            const range = maxLoss - minLoss || 1;
            
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            this.mlpLossHistory.forEach((loss, index) => {
                const x = (index / this.mlpLossHistory.length) * canvas.width;
                const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        }
        
        // Add labels
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        
        if (this.lossHistory.length > 0) {
            const minLoss = Math.min(...this.lossHistory);
            const maxLoss = Math.max(...this.lossHistory);
            ctx.fillStyle = '#007acc';
            ctx.fillText(`Simple Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 30);
            ctx.fillText(`Simple Max: ${maxLoss.toFixed(4)}`, 10, canvas.height - 15);
        }
        
        if (this.mlpLossHistory.length > 0) {
            const minLoss = Math.min(...this.mlpLossHistory);
            const maxLoss = Math.max(...this.mlpLossHistory);
            ctx.fillStyle = '#ff4444';
            ctx.fillText(`MLP Min: ${minLoss.toFixed(4)}`, 150, canvas.height - 30);
            ctx.fillText(`MLP Max: ${maxLoss.toFixed(4)}`, 150, canvas.height - 15);
        }
        
        // Add legend
        ctx.fillStyle = '#007acc';
        ctx.fillText('Simple Model', 300, 20);
        ctx.fillStyle = '#ff4444';
        ctx.fillText('MLP Deep Learning', 400, 20);
    }
    
    async visualizeEmbeddings() {
        const model = this.model || this.mlpModel;
        if (!model) return;
        
        this.updateStatus('Computing embedding visualization...');
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Sample items for visualization
            const sampleSize = Math.min(500, this.itemMap.size);
            const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
                Math.floor(i * this.itemMap.size / sampleSize)
            );
            
            // Get embeddings and compute PCA
            const embeddingsTensor = model.getItemEmbeddings();
            const embeddings = embeddingsTensor.arraySync();
            const sampleEmbeddings = sampleIndices.map(i => embeddings[i]);
            
            const projected = this.computePCA(sampleEmbeddings, 2);
            
            // Normalize to canvas coordinates
            const xs = projected.map(p => p[0]);
            const ys = projected.map(p => p[1]);
            
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            // Draw points
            ctx.fillStyle = 'rgba(0, 122, 204, 0.6)';
            sampleIndices.forEach((itemIdx, i) => {
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 40) + 20;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 40) + 20;
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            // Add title and labels
            ctx.fillStyle = '#000';
            ctx.font = '14px Arial';
            ctx.fillText('Item Embeddings Projection (PCA)', 10, 20);
            ctx.font = '12px Arial';
            ctx.fillText(`Showing ${sampleSize} items`, 10, 40);
            
            this.updateStatus('Embedding visualization completed.');
        } catch (error) {
            this.updateStatus(`Error in visualization: ${error.message}`);
        }
    }
    
    computePCA(embeddings, dimensions) {
        // Simple PCA using power iteration
        const n = embeddings.length;
        const dim = embeddings[0].length;
        
        // Center the data
        const mean = Array(dim).fill(0);
        embeddings.forEach(emb => {
            emb.forEach((val, i) => mean[i] += val);
        });
        mean.forEach((val, i) => mean[i] = val / n);
        
        const centered = embeddings.map(emb => 
            emb.map((val, i) => val - mean[i])
        );
        
        // Compute covariance matrix
        const covariance = Array(dim).fill(0).map(() => Array(dim).fill(0));
        centered.forEach(emb => {
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] += emb[i] * emb[j];
                }
            }
        });
        covariance.forEach(row => row.forEach((val, j) => row[j] = val / n));
        
        // Power iteration for first two components
        const components = [];
        for (let d = 0; d < dimensions; d++) {
            let vector = Array(dim).fill(1/Math.sqrt(dim));
            
            for (let iter = 0; iter < 10; iter++) {
                let newVector = Array(dim).fill(0);
                
                for (let i = 0; i < dim; i++) {
                    for (let j = 0; j < dim; j++) {
                        newVector[i] += covariance[i][j] * vector[j];
                    }
                }
                
                const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
                vector = newVector.map(val => val / norm);
            }
            
            components.push(vector);
            
            // Deflate the covariance matrix
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] -= vector[i] * vector[j];
                }
            }
        }
        
        // Project data
        return embeddings.map(emb => {
            return components.map(comp => 
                emb.reduce((sum, val, i) => sum + val * comp[i], 0)
            );
        });
    }
    
    async test() {
        if ((!this.model && !this.mlpModel) || this.qualifiedUsers.length === 0) {
            this.updateStatus('Model not trained or no qualified users found.');
            return;
        }
        
        this.updateStatus('Generating recommendations...');
        
        try {
            // Pick random qualified user
            const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const userInteractions = this.userTopRated.get(randomUser);
            const userIndex = this.userMap.get(randomUser);
            
            const results = {
                historical: userInteractions.slice(0, 10),
                simpleRecs: [],
                mlpRecs: []
            };
            
            // Get recommendations from simple model
            if (this.model) {
                const userEmb = this.model.getUserEmbedding(userIndex);
                const allItemScores = await this.model.getScoresForAllItems(userEmb);
                results.simpleRecs = this.filterAndSortRecommendations(randomUser, allItemScores);
            }
            
            // Get recommendations from MLP model
            if (this.mlpModel) {
                const userEmb = this.mlpModel.getUserEmbedding(userIndex);
                const allItemScores = await this.mlpModel.getScoresForAllItems(userEmb);
                results.mlpRecs = this.filterAndSortRecommendations(randomUser, allItemScores);
            }
            
            // Display results
            this.displayResults(randomUser, results);
            
        } catch (error) {
            this.updateStatus(`Error generating recommendations: ${error.message}`);
        }
    }
    
    filterAndSortRecommendations(userId, allItemScores) {
        const userInteractions = this.userTopRated.get(userId);
        const ratedItemIds = new Set(userInteractions.map(i => i.itemId));
        const candidateScores = [];
        
        allItemScores.forEach((score, itemIndex) => {
            const itemId = this.reverseItemMap.get(itemIndex);
            if (!ratedItemIds.has(itemId)) {
                candidateScores.push({ itemId, score, itemIndex });
            }
        });
        
        // Sort by score descending and take top 10
        candidateScores.sort((a, b) => b.score - a.score);
        return candidateScores.slice(0, 10);
    }
    
    displayResults(userId, results) {
        const resultsDiv = document.getElementById('results');
        
        let html = `<h2>Recommendations for User ${userId}</h2>`;
        
        if (this.currentModelType === 'both') {
            html += `<div class="three-column">`;
            
            // Historical ratings
            html += `<div><h3>Top 10 Rated Movies (Historical)</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Rating</th><th>Year</th></tr></thead><tbody>`;
            results.historical.forEach((interaction, index) => {
                const item = this.items.get(interaction.itemId);
                html += `<tr><td>${index + 1}</td><td>${item.title}</td><td>${interaction.rating}</td><td>${item.year || 'N/A'}</td></tr>`;
            });
            html += `</tbody></table></div>`;
            
            // Simple model recommendations
            html += `<div><h3>Simple Embedding Model</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Score</th><th>Year</th></tr></thead><tbody>`;
            results.simpleRecs.forEach((rec, index) => {
                const item = this.items.get(rec.itemId);
                html += `<tr><td>${index + 1}</td><td>${item.title}</td><td>${rec.score.toFixed(4)}</td><td>${item.year || 'N/A'}</td></tr>`;
            });
            html += `</tbody></table></div>`;
            
            // MLP model recommendations
            html += `<div><h3>MLP Deep Learning Model</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Score</th><th>Year</th></tr></thead><tbody>`;
            results.mlpRecs.forEach((rec, index) => {
                const item = this.items.get(rec.itemId);
                html += `<tr><td>${index + 1}</td><td>${item.title}</td><td>${rec.score.toFixed(4)}</td><td>${item.year || 'N/A'}</td></tr>`;
            });
            html += `</tbody></table></div>`;
            
            html += `</div>`;
        } else {
            html += `<div class="side-by-side">`;
            
            // Historical ratings
            html += `<div><h3>Top 10 Rated Movies (Historical)</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Rating</th><th>Year</th></tr></thead><tbody>`;
            results.historical.forEach((interaction, index) => {
                const item = this.items.get(interaction.itemId);
                html += `<tr><td>${index + 1}</td><td>${item.title}</td><td>${interaction.rating}</td><td>${item.year || 'N/A'}</td></tr>`;
            });
            html += `</tbody></table></div>`;
            
            // Model recommendations
            const modelRecs = this.currentModelType === 'simple' ? results.simpleRecs : results.mlpRecs;
            const modelName = this.currentModelType === 'simple' ? 'Simple Embedding Model' : 'MLP Deep Learning Model';
            
            html += `<div><h3>${modelName}</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Score</th><th>Year</th></tr></thead><tbody>`;
            modelRecs.forEach((rec, index) => {
                const item = this.items.get(rec.itemId);
                html += `<tr><td>${index + 1}</td><td>${item.title}</td><td>${rec.score.toFixed(4)}</td><td>${item.year || 'N/A'}</td></tr>`;
            });
            html += `</tbody></table></div>`;
            
            html += `</div>`;
        }
        
        resultsDiv.innerHTML = html;
        this.updateStatus('Recommendations generated successfully!');
    }
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
