using System.Text.Json;
using System.Text.Json.Nodes;
using Swan;

namespace Webscraper;

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using SpotifyAPI.Web;
using SpotifyAPI.Web.Auth;
using static PlaylistGen;
using static MeansShift;


public class SpotifyActor {
    
    public static async Task<string> GetAccessToken() {
            // Replace with your Spotify App credentials
            string clientId = "eba9a82feffa499d8511699526d6659c";
            string clientSecret = "75e616af5164409ca457ad119d6cac96";
            string authUrl = "https://accounts.spotify.com/api/token";
            


            var client = new HttpClient();
            var authData = new Dictionary<string, string> {
                { "grant_type", "client_credentials" }
            };

            var authRequest = new FormUrlEncodedContent(authData);
            client.DefaultRequestHeaders.Add("Authorization", "Basic " + Convert.ToBase64String(Encoding.UTF8.GetBytes(clientId + ":" + clientSecret)));

            var response = await client.PostAsync(authUrl, authRequest);
            var content = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode) {
                var authResult = JsonSerializer.Deserialize<Dictionary<string, Object>>(content);
                return ((JsonElement)authResult["access_token"]).GetString();
            } else {
                Console.WriteLine($"Error: {content}");
                return null;
            }
        }
        
        private static EmbedIOAuthServer _server;
        
        private static readonly string clientId = "eba9a82feffa499d8511699526d6659c";
        private static readonly string clientSecret = "75e616af5164409ca457ad119d6cac96";
        private static readonly string userId = "g47dvcltndgtgav7sgqsia10p";

        public static async Task authenticate(Func<object, AuthorizationCodeResponse, Task> callback) {
            // Make sure "http://localhost:5543/callback" is in your spotify application as redirect uri!
            _server = new EmbedIOAuthServer(new Uri("http://localhost:8887/callback"), 8887);
            await _server.Start();

            _server.AuthorizationCodeReceived += callback;
            _server.ErrorReceived += OnErrorReceived;

            var request = new LoginRequest(_server.BaseUri, clientId, LoginRequest.ResponseType.Code)
            {
                Scope = new List<string> { Scopes.UserReadEmail, Scopes.PlaylistReadPrivate, Scopes.PlaylistReadPrivate, Scopes.PlaylistModifyPublic }
            };
            BrowserUtil.Open(request.ToUri());
            Thread.Sleep(10000000);
        }

        private static async void createPlaylist(List<string> list, string description, SpotifyClient spotify)
        {
            if(list.Count == 0)
                return;

            StringBuilder descriptionBuilder = new StringBuilder(description);
            StringBuilder name = new StringBuilder(description);
            
            FullPlaylist playlist = await spotify.Playlists.Create(userId, new PlaylistCreateRequest(name.ToString()));
            playlist.Description = descriptionBuilder.ToString();
            
            for (int i = 0; i < list.Count / 100 + 1; i++) {
                int length = i * 100 + 100 < list.Count ? 100 : list.Count % 100;
                
                // Convert Track IDs to full Spotify URIs
                var trackUris = list.ConvertAll(id => $"spotify:track:{id}");
                
                PlaylistAddItemsRequest addItemsRequest = new PlaylistAddItemsRequest(trackUris.Slice(i * 100, length));
                SnapshotResponse snapshot = await spotify.Playlists.AddItems(playlist.Id, addItemsRequest); 
                Console.WriteLine(snapshot.Humanize());
            }
        }

        public static Dictionary<string, string> genreTable = new Dictionary<string, string>();

        private static async Task OnAuthorizationCodeReceived2(object sender, AuthorizationCodeResponse response)
        {
            await _server.Stop();

            var config = SpotifyClientConfig.CreateDefault();
            var tokenResponse = await new OAuthClient(config).RequestToken(
                new AuthorizationCodeTokenRequest(
                    clientId, clientSecret, response.Code, new Uri("http://localhost:8887/callback")
                )
            );

            var spotify = new SpotifyClient(tokenResponse.AccessToken);
            
            List<SongFeatureWrapper> songFeatures = ParsingTools.readHTMLIntoSongs("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\data.txt");
            List<SongFeatureWrapper> songs = ParsingTools.parseAudioFeatures("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\PCA-Sampling4Data.txt");
            
            // Cluster Songs
            (List<List<SongFeatureWrapper>> allClusters, List<float> distance) = KMeans(10, songs);

            int index = 0;
            foreach (var cluster in allClusters) {
                Console.WriteLine("Cluster: " + index + " - Count: " + cluster.Count + "TotalError: " + distance + "\n");
                foreach (var song in cluster) {
                    Console.WriteLine(song.id + "\n");
                }
            }
            
            // List<List<string>> lists = ParsingTools.songsToIDs(allClusters);
            // // Add Songs to Spotify Playlist
            // foreach (var list in lists)
            // {
            //     createPlaylist(list, "", spotify);
            // }
            //
            // // Write to File
            // string json = JsonSerializer.Serialize(lists);
            // File.WriteAllText("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\clusteredSongs.json", json);
            //
            // int count = 1;
            // string name = "All Liked Songs";
            // FullPlaylist playlist = await spotify.Playlists.Create(userId, new PlaylistCreateRequest(name));
            //
            // List<string> idList = songFeatures.Select(arr => arr.id).ToList();
            //
            // for (int i = 0; i < idList.Count / 100 + 1; i++) {
            //     int length = i * 100 + 100 < idList.Count ? 100 : idList.Count % 100;
            //         
            //     // Convert Track IDs to full Spotify URIs
            //     var trackUris = idList.ConvertAll(id => $"spotify:track:{id}");
            //         
            //     PlaylistAddItemsRequest addItemsRequest = new PlaylistAddItemsRequest(trackUris.Slice(i * 100, length));
            //     SnapshotResponse snapshot = await spotify.Playlists.AddItems(playlist.Id, addItemsRequest); 
            //     Thread.Sleep(20);
            //     Console.WriteLine(snapshot.ToString());
            // }

        }

        private static async Task OnErrorReceived(object sender, string error, string state)
        {
            Console.WriteLine($"Aborting authorization, error received: {error}");
            await _server.Stop();
        }

        public static async Task MakeLikedSongsPlaylist(object sender, AuthorizationCodeResponse response, List<string> songs, string description)
        {
            await _server.Stop();

            var config = SpotifyClientConfig.CreateDefault();
            var tokenResponse = await new OAuthClient(config).RequestToken(
                new AuthorizationCodeTokenRequest(
                    clientId, clientSecret, response.Code, new Uri("http://localhost:8887/callback")
                )
            );

            var spotify = new SpotifyClient(tokenResponse.AccessToken);

            createPlaylist(songs, description, spotify);
        }

        public static async Task OnAuthorizationCodeReceived(object sender, AuthorizationCodeResponse response)
        {
            await _server.Stop();

            var config = SpotifyClientConfig.CreateDefault();
            var tokenResponse = await new OAuthClient(config).RequestToken(
                new AuthorizationCodeTokenRequest(
                    clientId, clientSecret, response.Code, new Uri("http://localhost:8887/callback")
                )
            );

            var spotify = new SpotifyClient(tokenResponse.AccessToken);
            Dictionary<string, string> namesToIDs = ParsingTools.parseHTMLMakeDict("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\Data\\data.txt");

            List<SongFeatureWrapper> songs =
                ParsingTools.parseAudioFeatures(
                    "E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\Data\\PCA-SamplingRate2Variance90Data.txt");
            
            List<List<SongFeatureWrapper>> finalClusters = MeansShiftCluster(songs);

            List<List<string>> clusterIDs = new List<List<string>>();
            foreach (var cluster in finalClusters)
            {
                List<string> idList = new List<string>();
                foreach (var song in cluster)
                {
                    if (namesToIDs.ContainsKey(song.id))
                    { 
                        idList.Add(namesToIDs[song.id]);
                    }
                }
                clusterIDs.Add(idList);
            }
            
            List<List<string>> clusterNames = new List<List<string>>();
            foreach (var cluster in finalClusters)
            {
                List<string> nameList = new List<string>();
                foreach (var song in cluster)
                {
                    if (namesToIDs.ContainsKey(song.id))
                    { 
                        nameList.Add(song.id);
                    }
                }
                clusterNames.Add(nameList);
            }

            List<string> flattened = clusterNames.SelectMany(arr => arr).ToList();

            List<string> s = songs.Select(arr => arr.id).ToList();
            s.RemoveAll(arr => flattened.Contains(arr));

            foreach (var list in clusterIDs)
            {
                createPlaylist(list, "", spotify);
            }
            //Console.WriteLine("Total: " + bestError);
        }
}