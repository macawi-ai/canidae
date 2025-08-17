#!/usr/bin/env python3
"""
Simple WebSocket test server for CANIDAE WASM demo
"""

import asyncio
import json
import logging
from datetime import datetime
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connected pack members
pack_members = {}

async def handle_client(websocket, path):
    """Handle WebSocket connections from pack members"""
    member_info = None
    try:
        logger.info(f"New connection from {websocket.remote_address}")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type', 'message')
                
                if msg_type == 'join':
                    # Member joining the pack
                    pack_id = data.get('pack_id')
                    member_id = data.get('member_id')
                    member_info = {
                        'pack_id': pack_id,
                        'member_id': member_id,
                        'websocket': websocket
                    }
                    pack_members[member_id] = member_info
                    
                    logger.info(f"🐺 {member_id} joined pack {pack_id}")
                    
                    # Send welcome message
                    welcome = {
                        'type': 'welcome',
                        'message': f'Welcome to pack {pack_id}, {member_id}!',
                        'members': list(pack_members.keys()),
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(welcome))
                    
                    # Notify other pack members
                    notification = {
                        'type': 'member_joined',
                        'member_id': member_id,
                        'pack_id': pack_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    await broadcast_to_pack(pack_id, notification, exclude=member_id)
                    
                else:
                    # Regular message
                    logger.info(f"📦 Message from {data.get('member_id')}: {data.get('content')}")
                    
                    # Echo back with acknowledgment
                    response = {
                        'type': 'ack',
                        'original': data,
                        'timestamp': datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(response))
                    
                    # Broadcast to pack
                    if member_info and member_info.get('pack_id'):
                        await broadcast_to_pack(
                            member_info['pack_id'],
                            data,
                            exclude=member_info.get('member_id')
                        )
                        
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON: {message}")
                error_response = {
                    'type': 'error',
                    'message': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(error_response))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed for {member_info.get('member_id') if member_info else 'unknown'}")
    finally:
        # Remove member from pack
        if member_info and member_info.get('member_id'):
            member_id = member_info['member_id']
            if member_id in pack_members:
                del pack_members[member_id]
                logger.info(f"👋 {member_id} left the pack")
                
                # Notify other pack members
                notification = {
                    'type': 'member_left',
                    'member_id': member_id,
                    'timestamp': datetime.now().isoformat()
                }
                await broadcast_to_pack(
                    member_info.get('pack_id'),
                    notification
                )

async def broadcast_to_pack(pack_id, message, exclude=None):
    """Broadcast message to all pack members"""
    for member_id, member_info in pack_members.items():
        if member_info['pack_id'] == pack_id and member_id != exclude:
            try:
                await member_info['websocket'].send(json.dumps(message))
            except:
                logger.error(f"Failed to send to {member_id}")

async def main():
    """Start the WebSocket server"""
    logger.info("🐺 CANIDAE WebSocket Test Server")
    logger.info("Listening on ws://localhost:8080/pack")
    
    async with websockets.serve(handle_client, "localhost", 8080):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())