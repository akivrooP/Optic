import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, List
import time

app = FastAPI()

# In-memory storage for packets. Use a list to store recent packets.
# A more robust solution might use a proper queue or a database.
latest_packets: List[Dict[str, Any]] = []
MAX_PACKETS_STORED = 100 # Store up to 100 latest packets

class PacketData(BaseModel):
    packet: str # Base64 encoded packet string

@app.post("/send_packet")
async def send_packet(data: PacketData):
    """
    Receives base64 encoded packet data from the sender.
    """
    global latest_packets
    timestamp = time.time()
    packet_info = {"timestamp": timestamp, "packet": data.packet}
    
    # Add new packet and maintain the max size
    latest_packets.append(packet_info)
    if len(latest_packets) > MAX_PACKETS_STORED:
        latest_packets = latest_packets[-MAX_PACKETS_STORED:] # Keep only the latest
        
    print(packet_info)
        
    return {"message": "Packet received", "timestamp": timestamp}

@app.get("/get_latest_packets")
async def get_latest_packets(count: int = 1):
    """
    Returns the latest 'count' packets received.
    """
    global latest_packets
    # Return packets in chronological order
    return latest_packets[-count:]

@app.get("/")
async def read_root():
    return {"message": "Packet API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
