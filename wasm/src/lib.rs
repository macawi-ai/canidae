use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use web_sys::{WebSocket, MessageEvent, ErrorEvent, CloseEvent};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PackMessage {
    pub id: String,
    pub pack_id: String,
    pub member_id: String,
    pub content: String,
    pub timestamp: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PackMember {
    pub id: String,
    pub name: String,
    pub role: String,
    pub status: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PackConfig {
    pub pack_id: String,
    pub member_id: String,
    pub ws_url: String,
    pub heartbeat_interval: u32,
}

#[wasm_bindgen]
pub struct PackClient {
    config: PackConfig,
    ws: Option<WebSocket>,
    connected: bool,
    message_callback: Option<js_sys::Function>,
    error_callback: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl PackClient {
    #[wasm_bindgen(constructor)]
    pub fn new(pack_id: String, member_id: String, ws_url: String) -> Result<PackClient, JsValue> {
        console_log!("🐺 Initializing PackClient for pack: {}, member: {}", pack_id, member_id);
        
        Ok(PackClient {
            config: PackConfig {
                pack_id,
                member_id,
                ws_url,
                heartbeat_interval: 30000,
            },
            ws: None,
            connected: false,
            message_callback: None,
            error_callback: None,
        })
    }

    pub fn connect(&mut self) -> Result<(), JsValue> {
        console_log!("🔗 Connecting to: {}", self.config.ws_url);
        
        let ws = WebSocket::new(&self.config.ws_url)?;
        
        let ws_clone = ws.clone();
        let pack_id = self.config.pack_id.clone();
        let member_id = self.config.member_id.clone();
        
        let onopen = Closure::wrap(Box::new(move |_| {
            console_log!("✅ WebSocket connected for pack: {}, member: {}", pack_id, member_id);
            
            let join_msg = serde_json::json!({
                "type": "join",
                "pack_id": pack_id,
                "member_id": member_id,
                "timestamp": js_sys::Date::now(),
            });
            
            if let Ok(msg_str) = serde_json::to_string(&join_msg) {
                let _ = ws_clone.send_with_str(&msg_str);
            }
        }) as Box<dyn FnMut(JsValue)>);
        
        let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Ok(text) = e.data().dyn_into::<js_sys::JsString>() {
                console_log!("📦 Received message: {}", text);
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        
        let onerror = Closure::wrap(Box::new(move |e: ErrorEvent| {
            console_log!("❌ WebSocket error: {}", e.message());
        }) as Box<dyn FnMut(ErrorEvent)>);
        
        let onclose = Closure::wrap(Box::new(move |e: CloseEvent| {
            console_log!("🔌 WebSocket closed: code={}, reason={}", e.code(), e.reason());
        }) as Box<dyn FnMut(CloseEvent)>);
        
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
        
        onopen.forget();
        onmessage.forget();
        onerror.forget();
        onclose.forget();
        
        self.ws = Some(ws);
        self.connected = true;
        
        Ok(())
    }

    pub fn send_message(&self, content: String) -> Result<(), JsValue> {
        if !self.connected {
            return Err(JsValue::from_str("Not connected"));
        }
        
        let msg = PackMessage {
            id: format!("{}", js_sys::Date::now()),
            pack_id: self.config.pack_id.clone(),
            member_id: self.config.member_id.clone(),
            content,
            timestamp: js_sys::Date::now(),
        };
        
        let msg_str = serde_json::to_string(&msg)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        if let Some(ws) = &self.ws {
            ws.send_with_str(&msg_str)?;
            console_log!("📤 Sent message: {}", msg_str);
        }
        
        Ok(())
    }

    pub fn disconnect(&mut self) {
        console_log!("👋 Disconnecting PackClient");
        if let Some(ws) = &self.ws {
            let _ = ws.close();
        }
        self.connected = false;
        self.ws = None;
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn get_pack_id(&self) -> String {
        self.config.pack_id.clone()
    }

    pub fn get_member_id(&self) -> String {
        self.config.member_id.clone()
    }

    pub fn set_message_callback(&mut self, callback: js_sys::Function) {
        self.message_callback = Some(callback);
    }

    pub fn set_error_callback(&mut self, callback: js_sys::Function) {
        self.error_callback = Some(callback);
    }
}

#[wasm_bindgen]
pub fn init_canidae() -> Result<(), JsValue> {
    console_log!("🐺 CANIDAE WASM v1.1.0 initialized");
    console_log!("📦 Pack-oriented orchestration ready for browser");
    Ok(())
}

#[wasm_bindgen]
pub fn get_version() -> String {
    "1.1.0".to_string()
}