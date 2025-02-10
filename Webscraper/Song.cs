using System.Text;
using System.Text.Json.Serialization;

namespace Webscraper;


public class Song
{
    public static string[] FIELD_NAMES = new string[] { "popularity", "energy", "danceability", "happiness", "acousticness", "instrumentalness", "liveness", "speechiness", "loudness"};
    public Dictionary<string, int> features { get; set; }
    public string id { get; set; }
    public string key { get; set; }


    [JsonConstructor]
    public Song(string id, Dictionary<string, int> features, string key) {
        this.id = id;
        this.features = features;
        this.key = key;
    }

    public Song(string id) : this(id, null, null) {}

    public override string ToString() {
        StringBuilder sb = new StringBuilder();
        sb.Append("{");
        sb.Append("ID: " + id);
        sb.Append("[");
        foreach (var pair in features) {
            sb.Append(pair.Key + ": " + pair.Value + ", ");
        }
        sb.Append("]");
        sb.Append("}");
        return sb.ToString();
    }
}