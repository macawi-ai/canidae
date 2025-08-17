# 🐺 CANIDAE WASM Module

WebAssembly bindings for CANIDAE pack-oriented AI orchestration, enabling browser-based pack communication.

## Features

- **PackClient**: WebSocket-based pack member client
- **Real-time Messaging**: Send and receive pack messages
- **TypeScript Support**: Auto-generated TypeScript definitions
- **Zero Dependencies**: Pure WASM with minimal overhead
- **Browser-Native**: Runs directly in modern browsers

## Quick Start

### 1. Build the WASM Module

```bash
# Install wasm-pack if not already installed
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the WASM module
wasm-pack build --target web --out-dir pkg
```

### 2. Run the Demo

```bash
# Start both WebSocket and HTTP servers
./serve.sh

# Open http://localhost:8000 in your browser
```

### 3. Use in Your Application

```javascript
import init, { PackClient, init_canidae, get_version } from './pkg/canidae_wasm.js';

// Initialize WASM module
await init();
init_canidae();

// Create pack client
const client = new PackClient("arctic-pack", "synth", "ws://localhost:8080/pack");

// Set up callbacks
client.set_message_callback((msg) => {
    console.log("Received:", msg);
});

client.set_error_callback((err) => {
    console.error("Error:", err);
});

// Connect to pack
client.connect();

// Send messages
client.send_message("Hello pack!");

// Check connection status
if (client.is_connected()) {
    console.log("Connected to pack:", client.get_pack_id());
}

// Disconnect when done
client.disconnect();
```

## API Reference

### `PackClient`

Main class for pack communication.

#### Constructor
```typescript
new PackClient(pack_id: string, member_id: string, ws_url: string)
```

#### Methods

- `connect(): void` - Connect to the pack
- `disconnect(): void` - Disconnect from the pack
- `send_message(content: string): void` - Send a message to the pack
- `is_connected(): boolean` - Check connection status
- `get_pack_id(): string` - Get current pack ID
- `get_member_id(): string` - Get current member ID
- `set_message_callback(callback: Function): void` - Set message handler
- `set_error_callback(callback: Function): void` - Set error handler

### Global Functions

- `init_canidae(): void` - Initialize CANIDAE WASM
- `get_version(): string` - Get CANIDAE version

## Browser Compatibility

- Chrome 89+
- Firefox 87+
- Safari 14.1+
- Edge 89+

Requires WebAssembly and WebSocket support.

## Development

### Building
```bash
wasm-pack build --target web --out-dir pkg
```

### Testing
```bash
wasm-pack test --headless --firefox
```

### Size
- WASM: ~85KB (optimized with wasm-opt)
- JS Wrapper: ~15KB
- Total: ~100KB

## Integration with CANIDAE

This WASM module provides browser-based access to CANIDAE packs, enabling:

1. **Web Applications**: Build browser-based pack interfaces
2. **Real-time Dashboards**: Monitor pack activity in real-time
3. **Client-side Orchestration**: Run pack logic in the browser
4. **WebRTC Bridge**: Future support for P2P pack communication

## License

MIT - See LICENSE file in the root directory

---

Built with 🦊 by Synth & Cy