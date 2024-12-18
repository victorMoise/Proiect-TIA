import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from genres import genres, genre_popularity

N_SAMPLES = 1000

def generate_lego_dataset(n_samples=N_SAMPLES):
    data = []
    for _ in range(n_samples):
        piece_count = random.randint(50, 5000) # random piece count
        base_price_per_piece = random.uniform(0.08, 0.15) # random price per piece
        price = round(piece_count * base_price_per_piece, 2) # compute the price
        genre = random.choice(genres) # choose a random genre

        start_date = datetime(2000, 1, 1) 
        end_date = datetime(2024, 1, 1) 
        date_released = start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) # random release date between 2000 and 2024

        # calculate pieces sold
        units_sold = int((piece_count * genre_popularity[genre]) / (price + 1) * 10 + np.random.normal(0, 10)) 
        units_sold = max(1, units_sold)

        data.append([piece_count, date_released.strftime('%Y-%m-%d'), genre, genre_popularity[genre], price, round(base_price_per_piece, 4), units_sold])

    df = pd.DataFrame(data, columns=["piece_count", "date_released", "genre", 'genre_popularity', "price", "price_per_piece", "units_sold"])

    return df

dataset = generate_lego_dataset(n_samples=N_SAMPLES)
dataset.to_csv("lego_dataset.csv", index=False)
print("Dataset created and saved to 'lego_dataset.csv'.")