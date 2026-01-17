import os
import subprocess
from mcp.server.fastmcp import FastMCP

# ----------------------------
# CONFIG
# ----------------------------

MMAPP_PATH = r"C:\\MMWS\\MMMediaSuite\\MMApp\\Console\\Windows\\x64\Debug\\MMApp.exe"
OUTPUT_DIR = r"C:\\MMWS\\multimagix\\myconversion\\"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# MCP SERVER
# ----------------------------

mcp = FastMCP("MMApp Media Server")

@mcp.tool()
def mp4_to_mp3(input_file: str) -> str:
    """
    Convert an MP4 file to MP3 using MMApp.exe.
    Returns the output MP3 file path.
    """

    if not os.path.exists(input_file):
        return f"❌ Input file not found: {input_file}"
    
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    cmd = [
        MMAPP_PATH,
        "-f", "1",
        "-i", input_file,
        "-o", "1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False
        )

        if result.returncode != 0:
            return f"❌ MMApp failed:\n{result.stderr}"

        mp3_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".mp3")]

        if mp3_files:
            for f in mp3_files:
                output_file = f
        else:
            print("❌ No MP3 files found in the folder.")

        return f"✅ Conversion successful.\nMP3 saved at:\n{output_file}"

    except Exception as e:
        return f"❌ Error running MMApp: {str(e)}"


# ----------------------------
# START SERVER (stdio mode)
# ----------------------------

# if __name__ == "__main__":
#     mcp.run(transport="stdio")

if __name__ == "__main__":
    # Run as HTTP server on localhost:8000
    mcp.run(transport="sse")
