import time
import asyncio
import sys

def fetch_data_sync(source):
    print(f"Start fetching data from {source}...")
    time.sleep(2)  # Simulate an I/O operation with a 2-second delay
    print(f"Data fetched from {source}!")
    return {f"data from {source}": "sample data"}

async def fetch_data_async(source):
    print(f"Start fetching data from {source}...")
    await asyncio.sleep(2)  # Simulate an I/O operation with a 2-second delay
    print(f"Data fetched from {source}!")
    return {f"data from {source}": "sample data"}

def main_sync():
    print("Starting main function...")
    data1 = fetch_data_sync("source 1")
    data2 = fetch_data_sync("source 2")
    print(f"Received data: {data1}, {data2}")

async def main_async():
    print("Starting main function...")
    task1 = asyncio.create_task(fetch_data_async("source 1"))
    task2 = asyncio.create_task(fetch_data_async("source 2"))
    data1 = await task1
    data2 = await task2
    print(f"Received data: {data1}, {data2}")

if __name__ == "__main__":
    use_async = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else False
    if use_async:
        asyncio.run(main_async())
    else:
        main_sync()
