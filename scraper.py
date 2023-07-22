from nepse_scraper import Nepse_scraper
import pandas as pd
from datetime import date,timedelta

# create object from Nepse_scraper class
nepse = Nepse_scraper()

# getting data from nepse
with open('log','r') as log:
    last_date=log.read().split('-')
start_date = date(int(last_date[0]),int(last_date[1]),int(last_date[2])+1) 
end_date = date.today()    # perhaps date.now()
delta = end_date - start_date   # returns timedelta

for i in range(delta.days + 1):
    today = start_date + timedelta(days=i)
    data = nepse.get_today_price(today)
    df = pd.json_normalize(data['content'])
    df.to_csv(f'data.csv',mode='a',index=True)
    print(today)
    
with open('log','w') as log:
    log.write(end_date)


data = pd.read_csv('data.csv')
# Convert 'date' column to datetime type
data['businessDate'] = pd.to_datetime(data['businessDate'])

# Extract date features
data['year'] = data['businessDate'].dt.year
data['month'] = data['businessDate'].dt.month
data['day'] = data['businessDate'].dt.day
data['day_of_week'] = data['businessDate'].dt.dayofweek  # 0 for Monday, 6 for Sunday

# Handle cyclical features (month and day of the week)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# Drop the original date column
data = data.drop(columns=['businessDate'])