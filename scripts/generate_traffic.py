#!/usr/bin/env python3
"""
Global Web Traffic Generator
Genera tráfico HTTP/HTTPS realista visitando sitios de todo el mundo
Perfecto para poblar features de ML en detección de tráfico de red
"""

import requests
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sys

# Lista curada de sitios web de diferentes países y categorías
WEBSITES = {
    'news': [
        'https://www.bbc.com',
        'https://www.cnn.com',
        'https://www.reuters.com',
        'https://www.theguardian.com',
        'https://www.aljazeera.com',
        'https://www.lemonde.fr',
        'https://www.elpais.com',
        'https://www.spiegel.de',
        'https://www.timesofindia.com',
        'https://www.asahi.com',
    ],
    'tech': [
        'https://www.github.com',
        'https://stackoverflow.com',
        'https://news.ycombinator.com',
        'https://www.reddit.com',
        'https://techcrunch.com',
        'https://www.theverge.com',
        'https://arstechnica.com',
    ],
    'general': [
        'https://www.wikipedia.org',
        'https://www.amazon.com',
        'https://www.youtube.com',
        'https://www.twitter.com',
        'https://www.linkedin.com',
        'https://www.microsoft.com',
        'https://www.apple.com',
        'https://www.google.com',
    ],
    'education': [
        'https://www.mit.edu',
        'https://www.stanford.edu',
        'https://www.ox.ac.uk',
        'https://www.cam.ac.uk',
        'https://www.coursera.org',
        'https://www.edx.org',
    ],
    'regional': [
        'https://www.baidu.com',  # China
        'https://www.yandex.ru',  # Russia
        'https://www.naver.com',  # Korea
        'https://www.yahoo.co.jp',  # Japan
        'https://www.mercadolibre.com.ar',  # Argentina
        'https://www.globo.com',  # Brazil
        'https://www.bild.de',  # Germany
        'https://www.corriere.it',  # Italy
    ]
}

class TrafficGenerator:
    def __init__(self, duration_minutes=5, workers=10, delay_range=(0.5, 2.0)):
        self.duration = duration_minutes * 60  # Convert to seconds
        self.workers = workers
        self.delay_range = delay_range
        self.stats = {
            'requests': 0,
            'success': 0,
            'errors': 0,
            'bytes': 0
        }
        self.lock = threading.Lock()
        self.start_time = None

    def fetch_url(self, url, category):
        """Fetch a single URL and return stats"""
        try:
            # Random user agent
            user_agents = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            ]

            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }

            # Random timeout between 5-15 seconds
            timeout = random.uniform(5, 15)

            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)

            with self.lock:
                self.stats['requests'] += 1
                self.stats['success'] += 1
                self.stats['bytes'] += len(response.content)

            elapsed = time.time() - self.start_time
            print(f"[{elapsed:6.1f}s] ✅ {category:10s} | {response.status_code} | {len(response.content):8d}B | {url[:50]}")

            return True

        except requests.exceptions.Timeout:
            with self.lock:
                self.stats['requests'] += 1
                self.stats['errors'] += 1
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:6.1f}s] ⏱️  {category:10s} | TIMEOUT | {url[:50]}")
            return False

        except Exception as e:
            with self.lock:
                self.stats['requests'] += 1
                self.stats['errors'] += 1
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:6.1f}s] ❌ {category:10s} | ERROR   | {url[:50]} ({type(e).__name__})")
            return False

    def run(self):
        """Run the traffic generator"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║  Global Web Traffic Generator                              ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"\n⏱️  Duration: {self.duration/60:.1f} minutes")
        print(f"👷 Workers: {self.workers}")
        print(f"⏳ Delay range: {self.delay_range[0]:.1f}s - {self.delay_range[1]:.1f}s")
        print(f"🌍 Total websites: {sum(len(urls) for urls in WEBSITES.values())}")
        print(f"\n🚀 Starting traffic generation...\n")

        self.start_time = time.time()
        end_time = self.start_time + self.duration

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while time.time() < end_time:
                # Select random category
                category = random.choice(list(WEBSITES.keys()))
                url = random.choice(WEBSITES[category])

                # Submit task
                executor.submit(self.fetch_url, url, category)

                # Random delay between requests
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)

        self.print_stats()

    def print_stats(self):
        """Print final statistics"""
        elapsed = time.time() - self.start_time

        print("\n" + "="*60)
        print("📊 TRAFFIC GENERATION STATISTICS")
        print("="*60)
        print(f"⏱️  Duration:          {elapsed:.1f}s")
        print(f"📨 Total Requests:    {self.stats['requests']}")
        print(f"✅ Successful:        {self.stats['success']} ({self.stats['success']/max(self.stats['requests'],1)*100:.1f}%)")
        print(f"❌ Errors:            {self.stats['errors']} ({self.stats['errors']/max(self.stats['requests'],1)*100:.1f}%)")
        print(f"📦 Total Bytes:       {self.stats['bytes']:,} ({self.stats['bytes']/1024/1024:.2f} MB)")
        print(f"⚡ Request Rate:      {self.stats['requests']/elapsed:.2f} req/s")
        print(f"📈 Throughput:        {self.stats['bytes']/elapsed/1024:.2f} KB/s")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Parse arguments
    duration = 5  # minutes
    workers = 10

    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    if len(sys.argv) > 2:
        workers = int(sys.argv[2])

    # Run traffic generator
    generator = TrafficGenerator(
        duration_minutes=duration,
        workers=workers,
        delay_range=(0.3, 1.5)  # Faster than before for more traffic
    )

    try:
        generator.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        generator.print_stats()