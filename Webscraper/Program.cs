using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using RestSharp;
using System.Text.RegularExpressions;
using System.Net;
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json.Serialization;
using SpotifyAPI.Web;

namespace Webscraper {
    public class Webscraper
    {
        
        public static List<string> ReadInput() {
            string json = File.ReadAllText("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\Gorp-Songs.json");
            List<string> songIDs = JsonSerializer.Deserialize<List<string>>(json);
            return songIDs;
        }

        private async static Task<String?> sendRequest(string songID) {
            using (var client = new HttpClient()) {
                try {
                    // Set up the request URL and headers
                    string url = "http://localhost:3000/cf-clearance-scraper";
                    client.DefaultRequestHeaders.Add("HttpRequestMessage", "application/json");

                    // Create the request body
                    var requestBody = new {
                        url = "https://tunebat.com/Info/-XinU/" + songID,
                        mode = "source",
                        //ja3 = "772,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,23-27-65037-43-51-45-16-11-13-17513-5-18-65281-0-10-35,25497-29-23-24,0", // https://scrapfly.io/web-scraping-tools/ja3-fingerprint"
                        //userAgent = session.headers["user-agent"],
                    };

                    // Serialize the request body to JSON
                    string jsonBody = JsonSerializer.Serialize(requestBody, new JsonSerializerOptions { WriteIndented = true });

                    // Create the HttpContent (request body)
                    var content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

                    // Send the POST request
                    HttpResponseMessage response = await client.PostAsync(url, content);

                    // Ensure a successful status code (2xx)
                    response.EnsureSuccessStatusCode();

                    // Read and deserialize the JSON response
                    string responseBody = await response.Content.ReadAsStringAsync();

                    // Output the response
                    return responseBody;
                }
                catch (Exception ex)
                {
                    // Log any errors
                    Console.WriteLine("Error: " + ex.Message);
                    return null;
                }
            }
        }
        
        private async static Task<Song>? extractFeatures(string songID) {
            Dictionary<string, int> features = new Dictionary<string, int>();
            //$"https://api.scrapfly.io/scrape?tags=player%2Cproject%3Adefault&asp=true&render_js=false&proxy_pool=none&key=scp-live-854aa1978f494cffab0d86e6fd86beeb&url=https%3A%2F%2Ftunebat.com%2FInfo%2F-XinU%2F";
            
            Console.WriteLine("Retrieving features for: " + songID);
            try
            {

                String rawHTML = await sendRequest(songID);
                if (rawHTML == null)
                    return null;

                string generalFeaturesRegex = @"(?<=\\"" title=\\"")-?\d+(?= (db)?)";
                string keyRegex =
                    @"(?<=ant-typography\\""\\u003e).*?(?=\\u003c/h3\\u003e\\u003cspan class=\\""ant-typography ant-typography-secondary\\""\\u003ekey)";
                string bpmRegex =
                    @"(?<=ant-typography\\""\\u003e).*?(?=\\u003c/h3\\u003e\\u003cspan class=\\""ant-typography ant-typography-secondary\\""\\u003ebpm)";

                int index = 0;

                MatchCollection matches = Regex.Matches(rawHTML, generalFeaturesRegex);

                // Convert MatchCollection to an array of strings
                string[] matchedValues = matches.Cast<Match>()
                    .Select(m => m.Value)
                    .ToArray();

                foreach (var matchedValue in matchedValues)
                {
                    features[Song.FIELD_NAMES[index]] = int.Parse(matchedValue);
                    index++;
                }

                string key = "";
                Regex regex = new Regex(keyRegex, RegexOptions.Compiled);
                var match = regex.Match(rawHTML);
                if (match.Success)
                {
                    key = match.Value;
                }

                regex = new Regex(bpmRegex, RegexOptions.Compiled);
                match = regex.Match(rawHTML);
                if (match.Success)
                {
                    features[Song.FIELD_NAMES[index]] = int.Parse(match.Value);
                }

                index++;
                Console.WriteLine("Retrieved features for: " + songID);
                return new Song(songID, features, key);
            } catch (Exception e) {
                Console.WriteLine($"Error Retrieving features songID {songID}: {e.Message}");
                return null;
            }
        }

        public async static Task RequestAndProcess(List<string> allSongIDs) {
            ConcurrentBag<Song> allSongs = new ConcurrentBag<Song>();
            List<Task> tasks = new List<Task>();

            int count = 1;
            Semaphore limiter = new Semaphore(count, count);
            Random random = new Random();
            
            
            string filePath = "E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\allFeatures.json"; 
            await File.WriteAllTextAsync(filePath, string.Empty);

            StreamWriter writer = new StreamWriter(filePath, append: true);

            foreach (var id in allSongIDs) {
                limiter.WaitOne();
                tasks.Add(Task.Run(async () => {
                    int randomNumber = random.Next(20, 50);
                    
                    Task.Delay(randomNumber).Wait();
                    var features = await extractFeatures(id);
                    limiter.Release();
                    
                    if (features == null) {
                        await writer.WriteLineAsync($"ID: {id} was not retrieved.");
                        return;
                    }

                    allSongs.Add(features);
                    await writer.WriteLineAsync(features.ToString());
                }));
                
            }

            await Task.WhenAll(tasks);
        }

        public static string getSongFieldRegex(string fieldName) {
            string getSongFieldRegex = $"(?<={fieldName}: )\\d+(?=,)";
            return getSongFieldRegex;
        }

        public static Song[] readFromJSON(string filename) {
            if (File.Exists(filename)) {
                string entireFile = File.ReadAllText(filename);
                List<Song> allSongs = JsonSerializer.Deserialize<List<Song>>(entireFile);
                return allSongs.ToArray();
            }
            
            return null;
        }
        
        public static void transferToJSON(string filename) {
            string toPath = "E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\allFeatures.json"; 
            string fromPath = "E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\stuff.txt";
            string regex = "";
            
            List<Song> songs = new List<Song>();
            
            if (File.Exists(fromPath)) {
                string[] entireFile = File.ReadAllLines(fromPath);
                foreach (string line in entireFile) {
                    if (line[0] == '{') {
                        string id = line.Substring(5, 22);
                        string rest = line.Substring(23);
                        
                        Dictionary<string, int>features = new Dictionary<string, int>();
                        foreach (var field in Song.FIELD_NAMES) {
                            if (Regex.IsMatch(rest, getSongFieldRegex(field))) {
                                int value = int.Parse(Regex.Match(rest, getSongFieldRegex(field)).Value);
                                features[field] = value;
                            }
                        }
                        
                        Song songObject = new Song(id, features, "");
                        songs.Add(songObject);
                    }
                }
                
                string json = JsonSerializer.Serialize(songs);
                File.WriteAllText(toPath, json);
            }
        }
        
        /*
         * <tr class="google-visualization-table-tr-even"><td class="track-table-cell google-visualization-table-seq">3</td><td colspan="1" class="track-table-cell"><input class="track-select" type="checkbox" id="sel-1JSTJqkT5qHq8MDJnJbRE1" title="select to add this track to the staging list"></td><td colspan="1" class="track-table-cell"><span class="track-play glyphicon glyphicon-play" id="play-1JSTJqkT5qHq8MDJnJbRE1"></span></td><td colspan="1" class="track-table-cell">Every Breath You Take</td><td colspan="1" class="track-table-cell">The Police</td><td colspan="1" class="track-table-cell"></td><td colspan="1" class="track-table-cell">1983</td><td colspan="1" class="track-table-cell">2023‑10‑16</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">117</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">45</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">82</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">-10</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">7</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">74</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">254</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">54</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">3</td><td colspan="1" class="google-visualization-table-type-number track-table-cell">88</td></tr>
         */
        
        public static SongFeatureWrapper parseSongHTML(string html) {
            //Dictionary<string, int> features = new Dictionary<string, int>();
            string idRegex = "(?<=id=\"sel-).+?(?=\")";
            string dataRegex = "(?<=track-table-cell\">)[0-9a-zA-Z-‑ ]+?(?=<)";
            
            MatchCollection idMatches = Regex.Matches(html, idRegex);
            MatchCollection dataMatches = Regex.Matches(html, dataRegex);

            string id = idMatches[0].ToString();

            string[] metaData = new string[3];
            int[] features = new int[10];
            string genre = "";
            
            int count = 0;
            foreach (Match match in dataMatches) {
                string field = match.ToString();
                
                switch (count) {
                    case 0:
                    case 1:
                    case 3:
                    case 4:
                        metaData[0] = field;
                        break;
                    case 2:
                        genre = field;
                        break;
                    default:
                        features[count - 5] = int.Parse(field);
                        break;
                }
                count++;
            }
            
            
            
            SongFeatureWrapper song = new SongFeatureWrapper(id, features);
            return song;
        }

        public static List<SongFeatureWrapper> parseHTML(string filename) {
            string json = File.ReadAllText(filename);
            string regexOdd = "<tr class=\"google-visualization-table-tr-odd\">.+?</tr>";
            string regexEven = "<tr class=\"google-visualization-table-tr-even\">.+?</tr>";
            
            MatchCollection matchesOdd = Regex.Matches(json, regexOdd);
            MatchCollection matchesEven = Regex.Matches(json, regexEven);
            
            List<SongFeatureWrapper> allSongs = new List<SongFeatureWrapper>();
            
            foreach (Match matchedOdd in matchesOdd) {
                SongFeatureWrapper song = parseSongHTML(matchedOdd.ToString().ToString());
                allSongs.Add(song);
            }
            
            foreach (Match matchedEven in matchesEven) {
                SongFeatureWrapper song = parseSongHTML(matchedEven.ToString().ToString());
                allSongs.Add(song);
            }
            
            return allSongs;

        }

        public static List<List<string>> songsToIDs(List<List<SongFeatureWrapper>> clusters) {
            List<List<string>> ids = new List<List<string>>();

            foreach(var songList in clusters) {
                List<string> id = songList.Select(arr => arr.id).ToList();
                ids.Add(id);
            }
            
            return ids;
        }
        
        
        
        // public static async Task Main(string[] args) {
        //     await 
        // }
    }
}