namespace Webscraper;

using System.Linq;


public class PlaylistGen {
    private static double[] distanceToAllCentroids(int[] features, double[][] centroids) {
        double[] distances = new double[centroids.Length];
        for(int indexCentroid = 0; indexCentroid < centroids.Length; indexCentroid++) {
            double[] centroid = centroids[indexCentroid];
            
            double sumDistance = 0;
            for (int indexDimension = 0; indexDimension < features.Length; indexDimension++) {
                double difference = features[indexDimension] - centroid[indexDimension];
                sumDistance += difference * difference;
            }
            
            distances[indexCentroid] = Math.Sqrt(sumDistance);
        }
        
        return distances;
    }

    private static double[] calculateNewCentroid(int[][] features, int dimensions)
    {
        if (features.Length == 0)
            return null;
        
        double[] newCentroids = new double[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            var average = features
                .Select(arr => arr[i])        // Select the value at index `i` of each array
                .Average();                   // Calculate the average of those values
            newCentroids[i] = average;
        }
        
        return newCentroids;
    }

    // Using random partition
    private static double[][] initializeStartingCentroids(int num, int size) {
        double[][] centroids = new double[num][];
        Random rand = new Random();
        for (int count = 0; count < num; count++) {
            centroids[count] = Enumerable.Range(0, 10)  // Create a sequence with 10 elements
                .Select(x => (double)rand.Next(1, 101))  // Generate random numbers between 1 and 100
                .ToArray();
        }
        
        return centroids;
    }

    private static double[] calculateCentroidChange(double[][] centroids, double[][] newCentroids) {
        double[] totalChange = new double[newCentroids.Length];

        for (int indexCentroid = 0; indexCentroid < centroids.Length; indexCentroid++) {
            double change = 0;
            for (int indexDimension = 0; indexDimension < centroids[0].Length; indexDimension++) {
                change += Math.Abs(centroids[indexCentroid][indexDimension] - newCentroids[indexCentroid][indexDimension]);
            }
            totalChange[indexCentroid] = change;
        }
        
        return totalChange;
    }

    private static int minIndex(double[] values) {
        int minIndex = 0;

        for(int index = 0; index < values.Length; index++) {
            if (values[index] < values[minIndex]) {
                minIndex = index;
            }
        }
        
        return minIndex;
    }
    
    public static List<List<SongFeatureWrapper>> KMeans(int k, List<SongFeatureWrapper> songs) {
        int dimensions = songs[0].features.Length;
        double[][] centroids = initializeStartingCentroids(k, dimensions);
        
        List<List<SongFeatureWrapper>> clusters = new List<List<SongFeatureWrapper>>(k);
        
        bool converged = false;
        while(!converged) {
            clusters = new List<List<SongFeatureWrapper>>();
            for (int i = 0; i < k; i++) {
                clusters.Add(new List<SongFeatureWrapper>());
            }
            
            foreach(SongFeatureWrapper featureWrapper in songs) {
                double[] distances = distanceToAllCentroids(featureWrapper.features, centroids);
                int min = minIndex(distances);
                
                clusters[min].Add(featureWrapper);
            }
            
            double[][] newCentroids = new double[k][];
            for (int index = 0; index < clusters.Count; index++) {
                int[][] features = clusters[index].ToArray().Select(arr => arr.features).ToArray();
                double[] newCentroid = calculateNewCentroid(features, dimensions);
                if (newCentroid == null) {
                    newCentroid = centroids[index];
                }
                
                newCentroids[index] = newCentroid;

            }
            
            int[][] features2 = songs.ToArray().Select(arr => arr.features).ToArray();
            newCentroids = ReinitializeEmptyCentroids(features2, newCentroids, clusters);
            
            // Calculate convergence
            double[] change = calculateCentroidChange(centroids, newCentroids);
            if (change.Sum() < 1) {
                converged = true;
            }
            
            centroids = newCentroids;
        }
        
        return clusters;
    }
    
    public static double[][] ReinitializeEmptyCentroids(int[][] data, double[][] centroids, List<List<SongFeatureWrapper>> assignments)
    {
        int k = centroids.Length;
        Random rand = new Random();

        for (int i = 0; i < k; i++) {
            List<SongFeatureWrapper> cluster = assignments[i];
            if (cluster.Count == 0) { // If a centroid has no assigned points
                Console.WriteLine($"Reinitializing empty centroid {i}...");

                // Compute distances from each point to its nearest centroid
                double[] distances = data.Select(point => 
                    centroids.Min(centroid => EuclideanDistance(point.Select(arr => (double)arr).ToArray(), centroid))
                ).ToArray();

                // Compute probabilities (square the distances)
                double[] probabilities = distances.Select(d => d * d).ToArray();
                double sum = probabilities.Sum();

                // Normalize probabilities
                probabilities = probabilities.Select(p => p / sum).ToArray();

                // Select a new centroid with probability proportional to squared distance
                int newCentroidIndex = SampleFromDistribution(probabilities, rand);
                centroids[i] = data[newCentroidIndex].Select(arr => (double)arr).ToArray();
            }
        }

        return centroids;
    }

    private static double EuclideanDistance(double[] point1, double[] point2)
    {
        return Math.Sqrt(point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum());
    }

    private static int SampleFromDistribution(double[] probabilities, Random rand)
    {
        double cumulative = 0.0;
        double r = rand.NextDouble();
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (r <= cumulative)
                return i;
        }
        return probabilities.Length - 1; // Fallback
    }
}