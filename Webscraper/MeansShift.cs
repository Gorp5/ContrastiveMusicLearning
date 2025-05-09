namespace Webscraper;

public class MeansShift
{
    public static List<List<SongFeatureWrapper>> MeansShiftCluster(List<SongFeatureWrapper> songs)
    {
        float[][] data = songs.ToArray().Select(arr => arr.features).ToArray();
        
        List<float> flattened = data.ToList().SelectMany(x => x).ToList();
        float mean = flattened.Sum() / flattened.Count;
        
        List<float> differences = flattened.Select(x => x - mean).ToList();
        float squaredDifferences = differences.Select(x => x * x).Sum();
        float standardDeviation = (float)Math.Sqrt(squaredDifferences / differences.Count);
        

        Dictionary<string, List<SongFeatureWrapper>> clusters = new Dictionary<string, List<SongFeatureWrapper>>();
        for(int index = 0; index < data.Length; index++)
        {
            float[] point = data[index];
            bool converged = false;
            float[] meanCentroid = (float[])point.Clone();
            while (!converged)
            {
                float[] newMeanCentroid = shift(meanCentroid, data, gaussianKernel, standardDeviation);
                if (newMeanCentroid.Contains(Single.NaN))
                {
                    Console.WriteLine(1);
                }
                float dist = PlaylistGen.EuclideanDistance(meanCentroid, newMeanCentroid);
                meanCentroid = newMeanCentroid;
                if (dist == 0) {
                    converged = true;
                }
            }
            
            List<string> valuesAsStrings = meanCentroid.Select(x => x.ToString()).ToList();
            string key = string.Join("", valuesAsStrings);
            
            if (!clusters.ContainsKey(key)) {
                clusters.Add(key, new List<SongFeatureWrapper>());
            }
            
            clusters[key].Add(songs[index]);
        }
        
        return clusters.Values.ToList();
    }

    private static float flatKernel(float distance, float bandwidth, float dimensions)
    {
        if (distance > bandwidth) {
            return 0;
        } else {
            return 1;
        }
    }

    // Gaussian Kernel
    private static float gaussianKernel(float distance, float standardDeviation, float dimensions)
    {
        standardDeviation = 14f;
        if (Math.Abs(distance) < 1e-15) {
            return 0;
        }
        
        double numerator = Math.Pow(Math.E, -0.5f * Math.Pow(distance / standardDeviation, 2));
        double denominator = 4 * Math.Sqrt(2 * Math.PI) * standardDeviation;

        double gaussian = numerator / denominator;
        
        return (float)gaussian;
    }

    private static float[] shift(float[] meanCenter, float[][] data, Func<float, float, float, float> kernel, float bandwidth) {
        
        float[] totalShift = new float[meanCenter.Length];
        float scaleFactor = 0;
        
        foreach (var comparisonPoint in data) {
            float distance = PlaylistGen.EuclideanDistance(comparisonPoint, meanCenter);
            float weight = kernel(distance, bandwidth, meanCenter.Length);
            float[] shift = comparisonPoint.Select(arr => arr * weight).ToArray();
            totalShift = totalShift.Zip(shift, (a, b) => a + b).ToArray();
            scaleFactor += weight;
        }
        
        return totalShift.Select(arr => arr / scaleFactor).ToArray();
    }
}