import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from collections import defaultdict

def get_top_posts(subreddit, days=30, limit_per_day=25):
    base_url = f'https://old.reddit.com/r/{subreddit}/new/?sort=top&t=month'
    headers = {'User-Agent': 'Mozilla/5.0'}
    posts_by_day = defaultdict(list)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    while any(len(posts) < limit_per_day for posts in posts_by_day.values()) or len(posts_by_day) < days:
        response = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        for post in soup.find_all('div', class_='thing'):
            title = post.find('a', class_='title').text
            post_url = post.find('a', class_='title')['href']
            score = post.find('div', class_='score unvoted').text
            time_posted = datetime.fromtimestamp(int(post['data-timestamp']) // 1000)
            post_day = time_posted.strftime('%Y-%m-%d')

            if start_date.strftime('%Y-%m-%d') <= post_day <= end_date.strftime('%Y-%m-%d'):
                posts_by_day[post_day].append({'title': title, 'url': post_url, 'score': score, 'time_posted': time_posted})

        # Find the next page's URL
        next_button = soup.find('span', class_='next-button')
        if next_button:
            next_page_url = next_button.find('a')['href']
            base_url = next_page_url
        else:
            break  # No more pages to scrape

    # Trim each day's posts to the top 25
    for day in posts_by_day:
        posts_by_day[day] = sorted(posts_by_day[day], key=lambda x: x['score'], reverse=True)[:limit_per_day]

    return dict(posts_by_day)

subreddit = 'business'
top_posts = get_top_posts(subreddit)

# Print top posts sorted by day
print(f'Top 25 posts from the past month in r/{subreddit}, sorted by day:')
for day, posts in sorted(top_posts.items()):
    print(f'\nDay: {day}')
    for post in posts:
        print(f"{post['title']}")
        # print(f"{post['title']} - {post['url']} (Score: {post['score']})")
