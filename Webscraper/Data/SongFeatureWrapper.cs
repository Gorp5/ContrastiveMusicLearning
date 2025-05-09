using System.Text;

namespace Webscraper;

public class SongFeatureWrapper {
    public string id { get; set; }
    
    public float[] features { get; set; }

    public string name { get; set; }
    
    public string genre { get; set; }
    
    public string artist { get; set; }

    public SongFeatureWrapper(string id, float[] features) {
        this.id = id;
        this.features = features;
    }

    public SongFeatureWrapper(string id, float[] features, string artist, string name, string genre) : this(id, features)
    {
        this.name = name;
        this.genre = genre;
        this.artist = artist;
    }

    public override string ToString()
    {
        string feat = string.Join(", ", features.Select(f => f.ToString()));
        return $"\"{this.artist}\",\"{this.name}\",{this.id},{genre},\"[{feat}]\"";
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

    public string toCSVLine()
    {
        StringBuilder builder = new StringBuilder();
        foreach (float feature in features)
        {
            builder.Append(feature + " ");
        }
        
        builder.Append("\"" + id + "\"");
        
        return builder.ToString();
    }

    public string getReadableName()
    {
        return artist + " - " + name;
    }

    public SongFeatureWrapper(Song song) {
        this.id = song.id;
        features = song.features.Values.ToArray();
    }

    public class SongComparator : IEqualityComparer<SongFeatureWrapper>
    {
        public bool Equals(SongFeatureWrapper? s1, SongFeatureWrapper? s2)
        {
            if (ReferenceEquals(s1, s2))
                return true;

            if (s2 is null || s1 is null)
                return false;

            return s1.getReadableName().Equals(s2.getReadableName());
        }

        public int GetHashCode(SongFeatureWrapper s) => s.getReadableName().GetHashCode();
    }
}