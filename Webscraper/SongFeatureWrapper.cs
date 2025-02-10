namespace Webscraper;

public class SongFeatureWrapper {
    public string id { get; set; }
    
    public int[] features { get; set; }

    public SongFeatureWrapper(string id, int[] features) {
        this.id = id;
        this.features = features;
        this.features[0] = (int)(this.features[0] * 0.5f);     // Group BPM into high and low, and then reduce effect
        this.features[1] = (int)(this.features[1] * 1.75f);                            // Strengthen effect of Energy
        this.features[2] = (int)(this.features[2] * 1.5f);                            // Strengthen effect of Danceability
        this.features[3] = 0;                                                         // Not sure if Db helps group songs at all
        this.features[5] = (int)(this.features[5] * 1.5f);                            // Strengthen effect of Valence
        this.features[6] = (int)(sigmoidFeatures(features[0] - 150) * 0.02f);          // Group Duration into high and low, and then reduce effect
        this.features[9] = (int)(sigmoidFeatures(features[0]) * 0.02f);               // Group Popularity into high and low, and then reduce effect
    }

    private float sigmoidFeatures(float value) {
        double temp = -(value * 10 - 5);
        temp = Math.Pow(Math.E, temp);
        temp = 1 + temp;
        temp = 1 / temp;
        return (float) temp;
    }

    private float chainSigmoid(float value, int times) {
        if (times == 0) {
            return sigmoidFeatures(value);
        }
        
        return chainSigmoid(sigmoidFeatures(value), times - 1);
    }

    public SongFeatureWrapper(Song song) {
        this.id = song.id;
        features = song.features.Values.ToArray();
    }
}