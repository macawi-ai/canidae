# CANIDAE Mobile SDK

Mobile bindings for the CANIDAE AI orchestration platform, supporting iOS and Android through gomobile.

## Building the Mobile SDK

### Prerequisites

```bash
# Install gomobile
go install golang.org/x/mobile/cmd/gomobile@latest
go install golang.org/x/mobile/cmd/gobind@latest

# Initialize gomobile
gomobile init
```

### Build for iOS

```bash
# Build framework for iOS
gomobile bind -target=ios -o build/ios/Canidae.xcframework github.com/macawi-ai/canidae/pkg/client/mobile

# The output will be:
# - build/ios/Canidae.xcframework - iOS framework
```

### Build for Android

```bash
# Build AAR for Android
gomobile bind -target=android -o build/android/canidae.aar github.com/macawi-ai/canidae/pkg/client/mobile

# The output will be:
# - build/android/canidae.aar - Android library
# - build/android/canidae-sources.jar - Source files
```

## iOS Integration (Swift)

### 1. Add Framework to Xcode Project

1. Drag `Canidae.xcframework` into your Xcode project
2. Ensure it's added to your target's frameworks

### 2. Swift Example

```swift
import Canidae
import Foundation

class CanidaeManager {
    private let client: MobileCanidaeClient
    
    init() {
        client = MobileNewCanidaeClient()
    }
    
    func connect() async throws {
        // Configure client
        let config = [
            "serverEndpoint": "192.168.1.38:14001",
            "packID": "ios-pack",
            "apiKey": "your-api-key",
            "securityProfile": "enterprise"
        ]
        
        let configData = try JSONSerialization.data(withJSONObject: config)
        let configJSON = String(data: configData, encoding: .utf8)!
        
        try client.initialize(configJSON)
        try client.connect()
    }
    
    func executeAgent(prompt: String) async throws -> String {
        let request = MobileCreateExecuteRequest(
            "anthropic",
            prompt,
            "claude-3-opus",
            0.7,
            500
        )
        
        return try client.executeAgent(request)
    }
    
    func getStatus() throws -> [String: Any] {
        let statusJSON = try client.getStatus()
        let data = statusJSON.data(using: .utf8)!
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }
    
    deinit {
        try? client.close()
    }
}
```

### 3. SwiftUI View Example

```swift
import SwiftUI

struct CanidaeView: View {
    @StateObject private var viewModel = CanidaeViewModel()
    @State private var prompt = ""
    
    var body: some View {
        VStack {
            TextField("Enter prompt", text: $prompt)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button("Execute") {
                Task {
                    await viewModel.execute(prompt: prompt)
                }
            }
            .padding()
            
            if viewModel.isLoading {
                ProgressView()
            }
            
            if let response = viewModel.response {
                ScrollView {
                    Text(response)
                        .padding()
                }
            }
            
            if let error = viewModel.error {
                Text("Error: \(error)")
                    .foregroundColor(.red)
                    .padding()
            }
        }
        .onAppear {
            Task {
                await viewModel.connect()
            }
        }
    }
}
```

## Android Integration (Kotlin)

### 1. Add AAR to Android Project

In your `app/build.gradle`:

```gradle
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.aar'])
    // or
    implementation files('libs/canidae.aar')
}
```

### 2. Kotlin Example

```kotlin
import mobile.Mobile
import mobile.CanidaeClient
import kotlinx.coroutines.*
import org.json.JSONObject

class CanidaeManager {
    private val client: CanidaeClient = Mobile.newCanidaeClient()
    
    suspend fun connect() = withContext(Dispatchers.IO) {
        val config = JSONObject().apply {
            put("serverEndpoint", "192.168.1.38:14001")
            put("packID", "android-pack")
            put("apiKey", "your-api-key")
            put("securityProfile", "enterprise")
        }
        
        client.initialize(config.toString())
        client.connect()
    }
    
    suspend fun executeAgent(prompt: String): String = withContext(Dispatchers.IO) {
        val request = Mobile.createExecuteRequest(
            "anthropic",
            prompt,
            "claude-3-opus",
            0.7f,
            500
        )
        
        return@withContext client.executeAgent(request)
    }
    
    fun getStatus(): JSONObject {
        val statusJSON = client.getStatus()
        return JSONObject(statusJSON)
    }
    
    fun close() {
        client.close()
    }
}
```

### 3. Android Activity Example

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var canidaeManager: CanidaeManager
    private lateinit var binding: ActivityMainBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        canidaeManager = CanidaeManager()
        
        // Connect on startup
        lifecycleScope.launch {
            try {
                canidaeManager.connect()
                showToast("Connected to CANIDAE")
            } catch (e: Exception) {
                showToast("Connection failed: ${e.message}")
            }
        }
        
        // Setup execute button
        binding.executeButton.setOnClickListener {
            val prompt = binding.promptInput.text.toString()
            
            lifecycleScope.launch {
                try {
                    binding.progressBar.visibility = View.VISIBLE
                    val response = canidaeManager.executeAgent(prompt)
                    val responseData = JSONObject(response)
                    binding.responseText.text = responseData.getString("response")
                } catch (e: Exception) {
                    showToast("Execution failed: ${e.message}")
                } finally {
                    binding.progressBar.visibility = View.GONE
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        canidaeManager.close()
    }
    
    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
```

## React Native Integration

For React Native, use the native modules approach:

### iOS (Objective-C Bridge)

```objc
// CanidaeBridge.m
#import <React/RCTBridgeModule.h>
#import <Canidae/Canidae.h>

@interface CanidaeBridge : NSObject <RCTBridgeModule>
@property (nonatomic, strong) MobileCanidaeClient *client;
@end

@implementation CanidaeBridge

RCT_EXPORT_MODULE();

- (instancetype)init {
    if (self = [super init]) {
        self.client = MobileNewCanidaeClient();
    }
    return self;
}

RCT_EXPORT_METHOD(initialize:(NSString *)config
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
    NSError *error = nil;
    [self.client initialize:config error:&error];
    
    if (error) {
        reject(@"init_error", error.localizedDescription, error);
    } else {
        resolve(@YES);
    }
}

RCT_EXPORT_METHOD(executeAgent:(NSString *)request
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
    NSError *error = nil;
    NSString *response = [self.client executeAgent:request error:&error];
    
    if (error) {
        reject(@"exec_error", error.localizedDescription, error);
    } else {
        resolve(response);
    }
}

@end
```

### JavaScript Usage

```javascript
import { NativeModules } from 'react-native';

const { CanidaeBridge } = NativeModules;

class CanidaeClient {
  async initialize(config) {
    const configJSON = JSON.stringify(config);
    await CanidaeBridge.initialize(configJSON);
  }
  
  async executeAgent(request) {
    const requestJSON = JSON.stringify(request);
    const responseJSON = await CanidaeBridge.executeAgent(requestJSON);
    return JSON.parse(responseJSON);
  }
}

// Usage
const client = new CanidaeClient();

await client.initialize({
  serverEndpoint: '192.168.1.38:14001',
  packID: 'mobile-pack',
  apiKey: 'your-api-key',
  securityProfile: 'enterprise'
});

const response = await client.executeAgent({
  agent: 'anthropic',
  prompt: 'Hello, CANIDAE!',
  model: 'claude-3-opus',
  temperature: 0.7,
  maxTokens: 500
});
```

## Features

- **Simple JSON-based API**: All configuration and requests use JSON strings for easy integration
- **Thread-safe**: Safe to use from multiple threads/coroutines
- **Automatic reconnection**: Handles connection failures gracefully
- **Platform-optimized**: Optimized for mobile battery and network usage
- **Type-safe**: Strongly typed interfaces for Swift and Kotlin

## Performance Considerations

- **Connection pooling**: Reuse client instances
- **Batch requests**: Use chain operations for multiple agents
- **Background execution**: Use platform-specific background tasks
- **Caching**: Implement response caching where appropriate

## Security

- **API Key storage**: Use platform keychain/keystore
- **Certificate pinning**: Available through mTLS configuration
- **Secure transport**: All connections use TLS by default
- **Pack isolation**: Each mobile app gets its own pack

## Troubleshooting

### iOS Issues

- **Module not found**: Ensure framework is embedded in target
- **Simulator issues**: Build for `ios/amd64` for simulator support
- **Bitcode**: Disable bitcode if compilation fails

### Android Issues

- **ClassNotFoundException**: Ensure AAR is properly included
- **ProGuard**: Add keep rules for mobile package
- **MinSDK**: Requires Android API 21+

## Support

For issues or questions, see [CANIDAE Documentation](https://github.com/macawi-ai/canidae)