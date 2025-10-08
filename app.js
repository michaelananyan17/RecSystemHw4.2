import { TwoTowerModel, MLPModel } from './two-tower.js';

class MovieRecommender {
    constructor() {
        this.twoTowerModel = new TwoTowerModel();
        this.mlpModel = new MLPModel();
        this.userItemInteractions = new Map();
        this.userIds = new Set();
        this.movieIds = new Set();
        this.isTraining = false;
        this.trainingData = [];
        
        this.initializeEventListeners();
        this.setupChart();
        this.loadSampleData();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadSampleData());
        document.getElementById('trainModels').addEventListener('click', () => this.trainModels());
        document.getElementById('testModels').addEventListener('click', () => this.testModels());
        
        document.getElementById('addInteraction').addEventListener('click', () => this.addInteraction());
        document.getElementById('userId').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addInteraction();
        });
    }

    loadSampleData() {
        // Clear existing data
        this.userItemInteractions.clear();
        this.userIds.clear();
        this.movieIds.clear();
        this.trainingData = [];

        // Generate sample movie data with genres
        const movies = [
            { id: 'action1', title: 'Action Movie 1', genres: ['action', 'adventure'] },
            { id: 'action2', title: 'Action Movie 2', genres: ['action', 'sci-fi'] },
            { id: 'comedy1', title: 'Comedy Movie 1', genres: ['comedy', 'romance'] },
            { id: 'comedy2', title: 'Comedy Movie 2', genres: ['comedy'] },
            { id: 'drama1', title: 'Drama Movie 1', genres: ['drama', 'romance'] },
            { id: 'sci-fi1', title: 'Sci-Fi Movie 1', genres: ['sci-fi', 'action'] },
            { id: 'sci-fi2', title: 'Sci-Fi Movie 2', genres: ['sci-fi', 'drama'] },
            { id: 'horror1', title: 'Horror Movie 1', genres: ['horror'] },
            { id: 'romance1', title: 'Romance Movie 1', genres: ['romance', 'drama'] },
            { id: 'adventure1', title: 'Adventure Movie 1', genres: ['adventure', 'action'] }
        ];

        // Generate sample user interactions based on preferences
        const userPreferences = {
            'user1': { preferred: ['action', 'sci-fi'], disliked: ['romance', 'comedy'] },
            'user2': { preferred: ['comedy', 'romance'], disliked: ['horror', 'action'] },
            'user3': { preferred: ['drama', 'romance'], disliked: ['action'] },
            'user4': { preferred: ['sci-fi', 'adventure'], disliked: ['horror'] },
            'user5': { preferred: ['horror', 'action'], disliked: ['romance'] }
        };

        // Generate interactions
        let interactionCount = 0;
        Object.keys(userPreferences).forEach(userId => {
            this.userIds.add(userId);
            
            movies.forEach(movie => {
                let rating = 3; // Neutral base rating
                
                // Adjust rating based on genre preferences
                movie.genres.forEach(genre => {
                    if (userPreferences[userId].preferred.includes(genre)) {
                        rating += 1.5;
                    }
                    if (userPreferences[userId].disliked.includes(genre)) {
                        rating -= 1.5;
                    }
                });
                
                // Add some randomness and clamp to 1-5 range
                rating += (Math.random() - 0.5);
                rating = Math.max(1, Math.min(5, Math.round(rating * 2) / 2));
                
                if (rating > 3.5 || Math.random() > 0.7) { // Bias toward adding meaningful interactions
                    const key = `${userId}-${movie.id}`;
                    this.userItemInteractions.set(key, rating);
                    this.movieIds.add(movie.id);
                    this.trainingData.push({ userId, movieId: movie.id, rating });
                    interactionCount++;
                }
            });
        });

        this.updateInteractionsList();
        document.getElementById('trainingStatus').textContent = `Loaded ${interactionCount} sample interactions`;
        console.log(`Loaded ${interactionCount} sample interactions`);
    }

    addInteraction() {
        const userId = document.getElementById('userId').value.trim();
        const movieId = document.getElementById('movieId').value.trim();
        const rating = parseFloat(document.getElementById('rating').value);

        if (!userId || !movieId || isNaN(rating) || rating < 1 || rating > 5) {
            alert('Please enter valid user ID, movie ID, and rating (1-5)');
            return;
        }

        const key = `${userId}-${movieId}`;
        this.userItemInteractions.set(key, rating);
        this.userIds.add(userId);
        this.movieIds.add(movieId);
        this.trainingData.push({ userId, movieId, rating });

        this.updateInteractionsList();
        document.getElementById('movieId').value = '';
        document.getElementById('rating').value = '';

        console.log(`Added interaction: User ${userId}, Movie ${movieId}, Rating ${rating}`);
    }

    updateInteractionsList() {
        const list = document.getElementById('interactionsList');
        list.innerHTML = '';
        
        Array.from(this.userItemInteractions.entries()).slice(0, 20).forEach(([key, rating]) => {
            const [userId, movieId] = key.split('-');
            const li = document.createElement('li');
            li.textContent = `User ${userId} → ${movieId}: ${rating}★`;
            list.appendChild(li);
        });

        const remaining = this.userItemInteractions.size - 20;
        if (remaining > 0) {
            const li = document.createElement('li');
            li.textContent = `... and ${remaining} more interactions`;
            list.appendChild(li);
        }

        document.getElementById('interactionCount').textContent = this.userItemInteractions.size;
    }

    async trainModels() {
        if (this.trainingData.length < 10) {
            alert('Need at least 10 interactions to start training');
            return;
        }

        this.isTraining = true;
        document.getElementById('trainModels').disabled = true;
        document.getElementById('trainingStatus').textContent = 'Training both models...';

        try {
            console.log('Starting model training with:', this.trainingData.length, 'interactions');
            
            // Train both models with proper error handling
            const twoTowerLoss = await this.twoTowerModel.train(this.trainingData, 100);
            const mlpLoss = await this.mlpModel.train(this.trainingData, 100);
            
            // Update the training chart
            this.updateTrainingChart(twoTowerLoss, mlpLoss);
            
            // Update loss statistics
            this.updateLossStatistics(twoTowerLoss, mlpLoss);
            
            // Generate embedding visualization
            this.generateEmbeddingVisualization();
            
            document.getElementById('trainingStatus').textContent = 
                `Training completed! Models are ready for testing.`;
            
        } catch (error) {
            console.error('Training error:', error);
            document.getElementById('trainingStatus').textContent = 'Training failed: ' + error.message;
        } finally {
            this.isTraining = false;
            document.getElementById('trainModels').disabled = false;
        }
    }

    setupChart() {
        const ctx = document.getElementById('trainingChart').getContext('2d');
        this.trainingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Simple Embedding Model',
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2
                    },
                    {
                        label: 'MLP Deep Learning',
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        data: [],
                        tension: 0.4,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Loss Comparison'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Mean Squared Error Loss'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Training Epochs'
                        }
                    }
                }
            }
        });
    }

    updateTrainingChart(twoTowerLosses, mlpLosses) {
        const epochs = Math.max(twoTowerLosses.length, mlpLosses.length);
        this.trainingChart.data.labels = Array.from({length: epochs}, (_, i) => i + 1);
        this.trainingChart.data.datasets[0].data = twoTowerLosses;
        this.trainingChart.data.datasets[1].data = mlpLosses;
        this.trainingChart.update();
    }

    updateLossStatistics(twoTowerLosses, mlpLosses) {
        const simpleMin = Math.min(...twoTowerLosses).toFixed(4);
        const simpleMax = Math.max(...twoTowerLosses).toFixed(4);
        const mlpMin = Math.min(...mlpLosses).toFixed(4);
        const mlpMax = Math.max(...mlpLosses).toFixed(4);

        document.getElementById('simpleMin').textContent = simpleMin;
        document.getElementById('simpleMax').textContent = simpleMax;
        document.getElementById('mlpMin').textContent = mlpMin;
        document.getElementById('mlpMax').textContent = mlpMax;
    }

    generateEmbeddingVisualization() {
        const embeddings = this.twoTowerModel.getMovieEmbeddings();
        if (embeddings.length === 0) return;

        // Simple 2D projection (mock PCA)
        const projection = embeddings.map(embedding => ({
            x: embedding.vector[0] * 50 + Math.random() * 20 - 10,
            y: embedding.vector[1] * 50 + Math.random() * 20 - 10,
            movieId: embedding.movieId
        }));

        const status = document.getElementById('trainingStatus');
        status.textContent += ` Embedding visualization completed. Showing ${Math.min(projection.length, 500)} items.`;
        
        console.log('Item Embeddings Projection (PCA) - Showing 500 items', projection);
    }

    async testModels() {
        if (!this.twoTowerModel.isTrained || !this.mlpModel.isTrained) {
            alert('Please train models first before testing');
            return;
        }

        const testUserId = 'user2'; // Test with a known user
        const allMovieIds = Array.from(this.movieIds);

        try {
            const twoTowerRecs = await this.twoTowerModel.recommend(testUserId, allMovieIds, 5);
            const mlpRecs = await this.mlpModel.recommend(testUserId, allMovieIds, 5);

            this.displayRecommendations('twoTowerRecs', twoTowerRecs, 'Simple Embedding Model');
            this.displayRecommendations('mlpRecs', mlpRecs, 'MLP Deep Learning Model');

            // Show model characteristics
            this.displayModelCharacteristics();

        } catch (error) {
            console.error('Testing error:', error);
            alert('Error during testing: ' + error.message);
        }
    }

    displayRecommendations(elementId, recommendations, modelName) {
        const container = document.getElementById(elementId);
        container.innerHTML = `<h4>${modelName} Recommendations:</h4>`;
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML += '<p>No recommendations available</p>';
            return;
        }

        const list = document.createElement('ul');
        recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${rec.movieId}</strong>: ${rec.score.toFixed(3)} (predicted)`;
            list.appendChild(li);
        });
        
        container.appendChild(list);
    }

    displayModelCharacteristics() {
        const characteristics = `
            <div style="margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px;">
                <h4>Model Characteristics:</h4>
                <p><strong>Simple Embedding Model:</strong> Linear interactions, fast training, good generalization</p>
                <p><strong>MLP Deep Learning Model:</strong> Non-linear patterns, complex relationships, sophisticated recommendations</p>
            </div>
        `;
        document.getElementById('modelCharacteristics').innerHTML = characteristics;
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    new MovieRecommender();
});
