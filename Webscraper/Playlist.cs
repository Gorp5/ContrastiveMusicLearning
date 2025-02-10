using System.Text.Json;
using System.Text.Json.Nodes;

namespace Webscraper;

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using SpotifyAPI.Web;
using SpotifyAPI.Web.Auth;
using static PlaylistGen;


public class Playlist {
    public static async void makePlaylist(List<string> trackIds, string accessToken, int num) {

        if (!string.IsNullOrEmpty(accessToken)) {
            string userId = "g47dvcltndgtgav7sgqsia10p";

            // Create the playlist
            string playlistId = await CreatePlaylist(userId, accessToken, num);

            // Add tracks to the playlist
            if (!string.IsNullOrEmpty(playlistId)) {
                await AddTracksToPlaylist(playlistId, trackIds, accessToken);
                Console.WriteLine("Playlist created and tracks added successfully!");
            }
            else {
                Console.WriteLine("Failed to create playlist.");
            }
        }
    }
    
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

        private static async Task<string> CreatePlaylist(string userId, string accessToken, int num) {
            string url = $"https://api.spotify.com/v1/users/{userId}/playlists";

            var client = new HttpClient();
            //client.DefaultRequestHeaders.Add("Authorization", "Bearer " + accessToken);
            client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", accessToken);
            //client.DefaultRequestHeaders.Add("Content-Type", "application/json");


            // var body = new {
            //     name = $"Clustered Playlist {num}",
            //     description = $"Clustered Playlist #{num} made via K-Means Clustering",
            //     publicValue = true
            // };
            
            // JSON Payload
            string jsonPayload = @"{
                ""name"": $""Clustered Playlist {num}"",
                ""description"": $""Clustered Playlist #{num} made via K-Means Clustering"",
                ""public"": true
            }";

            //var json = JsonSerializer.Serialize(body)
            //content = new StringContent(json, Encoding.UTF8, "application/json");
            
            HttpContent content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            var response = client.PostAsync(url, content).GetAwaiter().GetResult();
            var responseContent = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode) {
                var result = JsonSerializer.Deserialize<Dictionary<string, string>>(responseContent);
                return result["id"];
            }else {
                Console.WriteLine($"Error: {responseContent}");
                return null;
            }
        }

        private static async Task AddTracksToPlaylist(string playlistId, List<string> trackIds, string accessToken)
        {
            string url = $"https://api.spotify.com/v1/playlists/{playlistId}/tracks";

            var client = new HttpClient();
            client.DefaultRequestHeaders.Add("Authorization", "Bearer " + accessToken);

            var body = new
            {
                uris = trackIds.ConvertAll(trackId => $"spotify:track:{trackId}")
            };
            
            var json = JsonSerializer.Serialize(body);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync(url, content);
            var responseContent = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                Console.WriteLine("Tracks added successfully!");
            }
            else
            {
                Console.WriteLine($"Error adding tracks: {responseContent}");
            }
        }
        
        private static EmbedIOAuthServer _server;
        
        private static readonly string clientId = "eba9a82feffa499d8511699526d6659c";
        private static readonly string clientSecret = "75e616af5164409ca457ad119d6cac96";
        private static readonly string userId = "g47dvcltndgtgav7sgqsia10p";

        public static async Task Main() {
            // Make sure "http://localhost:5543/callback" is in your spotify application as redirect uri!
            _server = new EmbedIOAuthServer(new Uri("http://localhost:8888/callback"), 8888);
            await _server.Start();

            _server.AuthorizationCodeReceived += OnAuthorizationCodeReceived;
            _server.ErrorReceived += OnErrorReceived;

            var request = new LoginRequest(_server.BaseUri, clientId, LoginRequest.ResponseType.Code)
            {
                Scope = new List<string> { Scopes.UserReadEmail, Scopes.PlaylistReadPrivate, Scopes.PlaylistReadPrivate, Scopes.PlaylistModifyPublic }
            };
            BrowserUtil.Open(request.ToUri());
            Thread.Sleep(10000000);
        }

        private static async Task OnAuthorizationCodeReceived(object sender, AuthorizationCodeResponse response)
        {
            await _server.Stop();

            var config = SpotifyClientConfig.CreateDefault();
            var tokenResponse = await new OAuthClient(config).RequestToken(
                new AuthorizationCodeTokenRequest(
                    clientId, clientSecret, response.Code, new Uri("http://localhost:8888/callback")
                )
            );

            var spotify = new SpotifyClient(tokenResponse.AccessToken);
            
            // do calls with Spotify and save token?
            
            //transferToJSON("output2.txt");
            //Song[] allSongs = readFromJSON("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\allFeatures.json");
            
            // Cluster Songs
            List<SongFeatureWrapper> songFeatures = Webscraper.parseHTML("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\data.txt");
            List<List<SongFeatureWrapper>> allClusters = KMeans(50, songFeatures);
            
            // Write to a file
            List<List<string>> lists = Webscraper.songsToIDs(allClusters);
            string json = JsonSerializer.Serialize(lists);
            File.WriteAllText("E:\\Coding\\SongAnalyzer\\Webscraper\\Webscraper\\clusteredSongs.json", json);

            int count = 1;
            
            foreach (List<string> list in lists.Slice(0, 6)) {
                if(list.Count == 0)
                    continue;
                
                FullPlaylist playlist = await spotify.Playlists.Create(userId, new PlaylistCreateRequest($"Generated {count}"));
                for (int i = 0; i < list.Count / 100 + 1; i++) {
                    int length = i * 100 + 100 < list.Count ? 100 : list.Count % 100;
                    
                    // Convert Track IDs to full Spotify URIs
                    var trackUris = list.ConvertAll(id => $"spotify:track:{id}");
                    
                    PlaylistAddItemsRequest addItemsRequest = new PlaylistAddItemsRequest(trackUris.Slice(i * 100, length));
                    SnapshotResponse snapshot = await spotify.Playlists.AddItems(playlist.Id, addItemsRequest); 
                    Console.WriteLine(snapshot.ToString());
                }
                count++;
            }
            
        }

        private static async Task OnErrorReceived(object sender, string error, string state)
        {
            Console.WriteLine($"Aborting authorization, error received: {error}");
            await _server.Stop();
        }
}