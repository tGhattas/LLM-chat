import asyncio
import os
from aiofiles import open as async_open


async def process_file(file_path):
    """ Process each file asynchronously. """
    try:
        async with async_open(file_path, mode='r') as file:
            content = await file.read()
            # Add your logic for processing the content here
            print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


async def crawl():
    root_path = '.'  # Change this to your target directory
    tasks = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Create a coroutine for each file and add it to the task list
            tasks.append(process_file(file_path))

    # Run all file processing tasks concurrently
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(crawl())
