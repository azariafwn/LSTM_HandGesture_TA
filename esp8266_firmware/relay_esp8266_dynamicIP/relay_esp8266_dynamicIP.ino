#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h> // <-- WAJIB: Tambahkan Library ini

const char* ssid = "zaaf";
const char* password = "12345678";

// GANTI NAMA INI UNTUK SETIAP ESP!
// ESP 1 -> "perangkat-1"
// ESP 2 -> "perangkat-2"
// const char* hostName = "perangkat-1"; 
const char* hostName = "perangkat-2"; 

ESP8266WebServer server(80);
const int relayPin = D1; 

// --- Handler Nyala (Universal) ---
// Fungsi ini akan dipanggil baik oleh perintah "11" maupun "21"
void handleRelayON() {
  digitalWrite(relayPin, LOW); // Nyalakan Relay (Active LOW)
  server.send(200, "text/plain", "Relay ON");
  Serial.println("Relay dinyalakan.");
}

// --- Handler Mati (Universal) ---
// Fungsi ini akan dipanggil baik oleh perintah "10" maupun "20"
void handleRelayOFF() {
  digitalWrite(relayPin, HIGH); // Matikan Relay
  server.send(200, "text/plain", "Relay OFF");
  Serial.println("Relay dimatikan.");
}

void handleRoot() {
  server.send(200, "text/plain", "ESP8266 Universal Node Siap!");
}


void setup() {
  Serial.begin(9600);
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, HIGH);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.println(WiFi.localIP());

  // --- BAGIAN DISCOVERY (BARU) ---
  if (MDNS.begin(hostName)) { 
    Serial.print("mDNS responder started: ");
    Serial.println(hostName);
    // Daftarkan service agar mudah dicari Python
    // Service: _http._tcp, Name: gesture-iot, Port: 80
    MDNS.addService("http", "tcp", 80);
    MDNS.addServiceTxt("http", "tcp", "type", "gesture-iot"); // Tag khusus kita
    // MDNS.addServiceTxt("http", "tcp", "id", "1"); // ID perangkat (Ganti jadi "2" di ESP kedua)
    MDNS.addServiceTxt("http", "tcp", "id", "2"); // ID perangkat (Ganti jadi "2" di ESP kedua)
  } else {
    Serial.println("Error setting up MDNS responder!");
  }
  // -------------------------------

  // --- ROUTING PINTAR (Universal) ---
  server.on("/", handleRoot);
  
  // ESP ini akan merespon jika dianggap sebagai "Perangkat 1"
  server.on("/11", handleRelayON); 
  server.on("/10", handleRelayOFF);
  
  // ESP ini JUGA akan merespon jika dianggap sebagai "Perangkat 2"
  server.on("/21", handleRelayON); 
  server.on("/20", handleRelayOFF);

  
  server.begin();
}

void loop() {
  server.handleClient();
  MDNS.update(); // <-- WAJIB: Update mDNS di loop
}