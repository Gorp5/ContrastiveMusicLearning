namespace Webscraper;

using System.Linq;


public class PlaylistGen {
    private static float[] DistanceToAllCentroids(float[] features, float[][] centroids) {
        float[] distances = new float[centroids.Length];
        for(int indexCentroid = 0; indexCentroid < centroids.Length; indexCentroid++) {
            float[] centroid = centroids[indexCentroid];

            distances[indexCentroid] = distanceBetweenPoints(centroid, features);
        }
        
        return distances;
    }

    private static float distanceBetweenPoints(float[] point1, float[] point2) {
        float sumDistance = 0;
        for (int indexDimension = 0; indexDimension < point2.Length; indexDimension++) {
            float difference = point2[indexDimension] - point1[indexDimension];
            sumDistance += difference * difference;
        }
            
        return (float)Math.Sqrt(sumDistance);
    }

    private static float[] CalculateNewCentroid(float[][] features, int dimensions)
    {
        if (features.Length == 0)
            return null;
        
        float[] newCentroids = new float[dimensions];
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
    private static float[][] sampleKRandomPoints(int k,  float[][] data) {
        float[][] centroids = new float[k][];
        Random rand = new Random();
        for (int count = 0; count < k; count++) {
            centroids[count] = data[rand.Next(0, data.Length)];
        }
        
        return centroids;
    }

    private static float[][] initializeKDistribution(int num, float[][] data)
    {
        List<float[]> centroids = new List<float[]>();
        
        Random rand = new Random();
        centroids.Add(data[rand.Next(0, data.Length)]);
        for (int index = 0; index < num - 1; index++) {
            centroids.Add(SelectCentroidFromDistribution(data, centroids.ToArray(), rand));
        }
        
        return centroids.ToArray();
    }

    private static float[] CalculateCentroidChange(float[][] centroids, float[][] newCentroids) {
        float[] totalChange = new float[newCentroids.Length];

        for (int indexCentroid = 0; indexCentroid < centroids.Length; indexCentroid++) {
            float change = 0;
            for (int indexDimension = 0; indexDimension < centroids[0].Length; indexDimension++) {
                change += Math.Abs(centroids[indexCentroid][indexDimension] - newCentroids[indexCentroid][indexDimension]);
            }
            totalChange[indexCentroid] = change;
        }
        
        return totalChange;
    }

    private static int MinIndex(float[] values) {
        int minIndex = 0;

        for(int index = 0; index < values.Length; index++) {
            if (values[index] < values[minIndex]) {
                minIndex = index;
            }
        }
        
        return minIndex;
    }
    
    public static (List<List<SongFeatureWrapper>> clusters, List<float> totalDistance) KMeans(int k, List<SongFeatureWrapper> songs) {
        int dimensions = songs[0].features.Length;
        float[][] allData = songs.ToArray().Select(arr => arr.features).ToArray();
        float[][] centroids = initializeKDistribution(k, allData);//sampleKRandomPoints(k, allData);
        
        List<List<SongFeatureWrapper>> clusters = new List<List<SongFeatureWrapper>>(k);
        
        bool converged = false;
        while(!converged) {
            clusters = new List<List<SongFeatureWrapper>>();
            for (int i = 0; i < k; i++) {
                clusters.Add(new List<SongFeatureWrapper>());
            }
            
            foreach(SongFeatureWrapper featureWrapper in songs) {
                float[] distances = DistanceToAllCentroids(featureWrapper.features, centroids);
                int min = MinIndex(distances);
                
                clusters[min].Add(featureWrapper);
            }
            
            float[][] newCentroids = new float[k][];
            for (int index = 0; index < clusters.Count; index++) {
                float[][] features = clusters[index].ToArray().Select(arr => arr.features).ToArray();
                float[] newCentroid = CalculateNewCentroid(features, dimensions);
                if (newCentroid == null) {
                    newCentroid = centroids[index];
                }
                
                newCentroids[index] = newCentroid;

            }
            
            float[][] features2 = songs.ToArray().Select(arr => arr.features).ToArray();
            newCentroids = ReinitializeEmptyCentroids(features2, newCentroids, clusters);
            
            // Calculate convergence
            float[] change = CalculateCentroidChange(centroids, newCentroids);
            if (change.Sum() < 1) {
                converged = true;
            }
            
            centroids = newCentroids;
        }
        
        List<float> totalDistance = new List<float>();
        
        for (int index = 0; index < centroids.Length; index++) {
            float tempDistance = 0;
            foreach (var song in clusters[index]) {
                float distance = distanceBetweenPoints(centroids[index], song.features);
                tempDistance += Math.Abs(distance);
            }
            
            totalDistance.Add(tempDistance);
        }
        
        return (clusters, totalDistance);
    }
    
    public static float[] SelectCentroidFromDistribution(float[][] data, float[][] centroids, Random rand)
    {
        // Compute distances from each point to its nearest centroid
        float[] distances = data.Select(point => 
            centroids.Min(centroid => EuclideanDistance(point, centroid))
        ).ToArray();

        // Compute probabilities (square the distances)
        float[] probabilities = distances.Select(d => d * d).ToArray();
        float sum = probabilities.Sum();

        // Normalize probabilities
        probabilities = probabilities.Select(p => p / sum).ToArray();

        // Select a new centroid with probability proportional to squared distance
        int newCentroidIndex = SampleFromDistribution(probabilities, rand);
        return data[newCentroidIndex];
    }
    
    public static float[][] ReinitializeEmptyCentroids(float[][] data, float[][] centroids, List<List<SongFeatureWrapper>> assignments)
    {
        int k = centroids.Length;
        Random rand = new Random();

        for (int i = 0; i < k; i++) {
            List<SongFeatureWrapper> cluster = assignments[i];
            if (cluster.Count == 0) { // If a centroid has no assigned points
                Console.WriteLine($"Reinitializing empty centroid {i}...");
                centroids[i] = SelectCentroidFromDistribution(data, centroids, rand);
            }
        }

        return centroids;
    }

    public static float EuclideanDistance(float[] point1, float[] point2)
    {
        return (float)Math.Sqrt(point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum());
    }

    private static int SampleFromDistribution(float[] probabilities, Random rand)
    {
        float cumulative = 0.0f;
        float r = rand.NextSingle();
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (r <= cumulative)
                return i;
        }
        return probabilities.Length - 1; // Fallback
    }
}