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
using System.Runtime.Loader;
using System.Text;
using System.Text.Json.Serialization;
using SpotifyAPI.Web;
using SpotifyAPI.Web.Http;

namespace Webscraper {
    public class ParsingTools
    {
        // TODO: Parse using XML
        public static SongFeatureWrapper parseSongHTML(string html) {
            //Dictionary<string, int> features = new Dictionary<string, int>();
            string idRegex = "(?<=id=\"sel-).+?(?=\")";
            string dataRegexReplace = "<.+?>";
            string filteredDataRegex = "[^|]+";
            string dataRegex = "(?<=track-table-cell\">)[0-9a-zA-Z-‑ ]+?(?=<)";

            string replaced = Regex.Replace(html, dataRegexReplace, "|");
            
            MatchCollection dataMatches = Regex.Matches(replaced, filteredDataRegex);
            MatchCollection idMatches = Regex.Matches(html, idRegex);
            string[] data = replaced.Split('|');
            data = data.Where(arr => arr != "").ToArray();
            
            string artist = data[2];
            string title = data[1];

            string id = idMatches[0].ToString();

            string[] metaData = new string[3];
            float[] features = new float[10];
            string genre = "";
            
            int count = 0;
            bool flag = false;
            foreach (Match match in dataMatches) {
                string field = match.ToString();
                // if (flag) {
                //     if(!int.TryParse(field, out _)) {
                //         genre = field;
                //     }
                //     else
                //     {
                //         genre = "No Labeled Genre";
                //     }
                //     
                //     flag = false;
                // }
                
                if (count == 3) {
                    if(dataMatches.Count == 15) {
                        metaData[0] = field;
                        count+=2;
                        flag = true;
                        genre = "No Labeled Genre";
                        continue;
                    }
                    
                    genre = field;
                }
                
                switch (count) {
                    case 0:
                    case 1:
                    case 2:
                    case 4:
                    case 5:
                        metaData[0] = field;
                        break;
                    case 3:
                        genre = field;
                        break;
                    default:
                        string formatted = field.Replace(",", "");
                        if(formatted == "NaN")
                            continue;
                        
                        features[count - 6] = int.Parse(formatted);
                        break;
                }
                count++;
            }
            
            // SpotifyActor.genreTable[id] = genre;
            
            SongFeatureWrapper song = new SongFeatureWrapper(id, features, artist, title, genre);
            return song;
        }

        public static (string filetitle, string id) parseHTMLDictionary(string html)
        {
            string idRegex = "(?<=id=\"sel-).+?(?=\")";
            string dataRegexReplace = "<.+?>";

            string replaced = Regex.Replace(html, dataRegexReplace, "|");
             
            MatchCollection idMatches = Regex.Matches(html, idRegex);
        
            string[] data = replaced.Split('|');
            data = data.Where(arr => arr != "").ToArray();
            
            string fileTitle = data[2] + " - " + data[1];
            string id = idMatches[0].ToString();

            return (fileTitle, id);
        }

        public static Dictionary<string, string> parseHTMLMakeDict(string filename)
        {
            string json = File.ReadAllText(filename);
            string regexOdd = "<tr class=\"google-visualization-table-tr-odd\">.+?</tr>";
            string regexEven = "<tr class=\"google-visualization-table-tr-even\">.+?</tr>";
            
            MatchCollection matchesOdd = Regex.Matches(json, regexOdd);
            MatchCollection matchesEven = Regex.Matches(json, regexEven);
            
            Dictionary<string, string> nameToID = new Dictionary<string, string>();
            
            foreach (Match matchedOdd in matchesOdd) {
                (string title, string id) = parseHTMLDictionary(matchedOdd.ToString());
                if (!nameToID.ContainsKey(title))
                {
                    nameToID.Add(title, id);
                }
            }
            
            foreach (Match matchedEven in matchesEven) {
                (string title, string id) = parseHTMLDictionary(matchedEven.ToString());
                if (!nameToID.ContainsKey(title))
                {
                    nameToID.Add(title, id);
                }
            }
            
            return nameToID;
        }

        public static List<SongFeatureWrapper> readHTMLIntoSongs(string filename) {
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
        
        
        public static List<string> songsToIDs(List<SongFeatureWrapper> songList) {
            return songList.Select(arr => arr.id).ToList();
        }
        
        public static Dictionary<string, string> songsToIDMap(List<SongFeatureWrapper> songList) {
            return songList.ToDictionary(arr => arr.getReadableName(), arr => arr.id);
        }

        public static List<SongFeatureWrapper> parseAudioFeatures(string filename)
        {
            List<string> allLines = File.ReadAllLines(filename).ToList();

            List<SongFeatureWrapper> allSongs = new List<SongFeatureWrapper>();
            foreach (var line in allLines)
            {
                string[] pieces = line.Split(": ");
                string name = pieces[0].Substring(2);
                name = name.Remove(name.Length - 1, 1);
                if(name.Split(" - ")[0].Split(",").Length > 1)
                {
                    string[] names = name.Split(" - ");
                    name = name.Split(",")[0] + " - " + names[1];
                }
                string[] featureStrings = pieces[1].Split(",");
                List<float> features = new List<float>();

                foreach (var featureString in featureStrings)
                {
                    string cleanedFeature = featureString.Replace("[", "");
                    cleanedFeature = cleanedFeature.Replace("]", "");
                    cleanedFeature = cleanedFeature.Replace("}", "");
                    cleanedFeature = cleanedFeature.Replace("{", "");

                    features.Add(float.Parse(cleanedFeature));
                }

                
                SongFeatureWrapper song = new SongFeatureWrapper(name, features.ToArray());
                allSongs.Add(song);
            }

            return allSongs;
        }
        
        public static List<string> parseClusterFile(string filename, int featureCount, int metadataLines)
        {
            List<string> allLines = File.ReadAllLines(filename).ToList();
            
            // Fisrt 9 lines are not relevant
            allLines = allLines.GetRange(metadataLines, allLines.Count - metadataLines);
            int extensionNum = 4;
            
            List<string> allSongs = new List<string>();
            foreach (var line in allLines)
            {
                int featureSegments = featureCount + 1;
                List<string> lineSegments = line.Split(" ").ToList();
                string name = string.Join(" ", lineSegments.GetRange(featureSegments, lineSegments.Count - featureSegments));
                name = name.Remove(name.Length - extensionNum, extensionNum);
                
                allSongs.Add(name);
            }

            return allSongs;
        }
        
        public static async Task Main(string[] args)
        {
            string htmlData = "\\Data\\data3-10-25.txt";
            string clusterData = "\\output_analysis\\elki-FINAL-Linear-252-COS\\clu_0.02331035207414631.txt";
            List<SongFeatureWrapper> songs = readHTMLIntoSongs(htmlData);
            
            songs = songs.Distinct(new SongFeatureWrapper.SongComparator()).ToList();
            Dictionary<string, string> songIds = songsToIDMap(songs);
            
            List<string> clusters = parseClusterFile(clusterData, 512, 10);
            
            List<string> clusterIDs = clusters.Where(arr => songIds.Keys.Contains(arr)).Select(arr => songIds[arr]).ToList();
            
            await SpotifyActor.authenticate((sender, response) => SpotifyActor.MakeLikedSongsPlaylist(sender, response, clusterIDs, "Cluster 1"));
            
        }
    }
}