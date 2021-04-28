---
title: Sentiment Analysis using Naive Bayes Classifier
subtitle: Implimenting Naive Bayes classifier from scratch for sentiment analysis of Yelp dataset

# Summary for listings and search engines
summary: Implimenting Naive Bayes classifier from scratch for sentiment analysis of Yelp dataset.

# Link this post with a project
projects: []

# Date published
date: "2021-04-27T00:00:00Z"

# Date updated
lastmod: "2021-04-27T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'https://miro.medium.com/max/1200/1*ZW1icngckaSkivS0hXduIQ.jpeg'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- data mining

categories:
- blog
---

##  

{{< hl >}} _Switch to dark mode for better readability_ {{< /hl >}}

[_Link to Jupyter Notebook_](https://github.com/rohanmandrekar/Sentiment-Analysis/blob/main/Sentiment_Analysis.ipynb)

### What is Bayes Theorem?

Bayes theorem named after Rev. Thomas Bayes. It works on conditional probability. Conditional probability is the probability that something will happen, given that something else has already occurred. Using the conditional probability, we can calculate the probability of an event using its prior knowledge. [_Source_](https://dataaspirant.com/naive-bayes-classifier-machine-learning/#:~:text=Naive%20Bayes%20is%20a%20kind,as%20the%20most%20likely%20class.)

Below is the formula for conditional probability using Bayes Theorem:
![png](./bayes_rule.png)
[_Source_](https://www.analyticsvidhya.com/wp-content/uploads/2015/09/Bayes_rule-300x172.png)

### What is the relevance of each word in 'Naive Bayes Classifier'?

**Naive:** The word 'Naive' indicates that the algorithm assumes independence among attributes Xi when class is given: P(X1, X2, ..., Xd|Yj) = P(X1| Yj) P(X2| Yj)... P(Xd| Yj)
**Bayes:** This signifies the use of Bayes theorem to calculate conditional probablity
**Classifier:** Shows that the application of the algorithm is to classify a given set of inputs

### What is Laplace Smoothing?
If one of the conditional probabilities is zero, then the entire expression becomes zero. To solve this error we use Lapace Smoothing. To perform Laplace smoothing we add 1 to the numerator and 'v' to the denomenator of all probabilites. where 'v' is the total number of attribute values that Xi can take


### Accuracy on test dataset before smoothening: 54%
### Accuracy on test dataset after performing laplacian smoothening: 68%



### Challenges faced:
Initialy when I attempted to implement the classifier on the IMDB dataset, only 768 out of 1000 lines were being read. I attempted to fix it, but did not succeed. Eventually I switched to the Yelp dataset, and the issue was resolved.


### My Observations and Experiments:
I tried eleminating a few stop words from the data like 'the','a','and','of','is','to','this','was','in','that','it','for','as','with','are','on', and 'i' but this showed no change in the accuracy of the classifier.

### References at the end of the page



```python
import pandas as pd
import numpy as np
```


```python
from google.colab import drive
drive.mount("/content/drive/")
```

    Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount("/content/drive/", force_remount=True).
    


```python
path='/content/drive/My Drive/Colab Notebooks/imdb/yelp.txt'
```


```python
df = pd.read_csv(path, names=['sentence', 'label'], delimiter='\t',header=None)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentence</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(len(df))
```

    1000
    

# Split data into train, validation and test


```python
#reference: https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-in-training-validation-and-test-set#:~:text=To%20split%20into%20train%20%2F%20validation%20%2F%20test,split%20by%20calling%20scikit-learn%27s%20function%20train_test_split%20%28%29%20twice.

fractions=np.array([0.8,0.1,0.1])
df=df.sample(frac=1)
train_set, val_set, test_set = np.array_split(
    df, (fractions[:-1].cumsum() * len(df)).astype(int))
```


```python
print('length of training data: ',len(train_set))
print(train_set.head())
print('length of validation data: ',len(val_set))
print(val_set.head())
print('length of testing data: ',len(test_set))
print(test_set.head())

```

    length of training data:  800
                                                  sentence  label
    702  Have been going since 2007 and every meal has ...      1
    723  Special thanks to Dylan T. for the recommendat...      1
    409                               TOTAL WASTE OF TIME.      0
    998  The whole experience was underwhelming, and I ...      0
    555  I know this is not like the other restaurants ...      0
    length of validation data:  100
                                                  sentence  label
    458                  Best tater tots in the southwest.      1
    331  Both of them were truly unbelievably good, and...      1
    156  this was a different cut than the piece the ot...      1
    239  Everyone is very attentive, providing excellen...      1
    827  For that price I can think of a few place I wo...      0
    length of testing data:  100
                                                  sentence  label
    651  Great place to relax and have an awesome burge...      1
    947                          I was VERY disappointed!!      0
    465  The food was outstanding and the prices were v...      1
    444  Each day of the week they have a different dea...      1
    142  My husband and I ate lunch here and were very ...      0
    


```python
from collections import Counter
```


```python
positive_count = Counter()
negative_count = Counter()
total_count = Counter()

train_list=train_set.values.tolist()
```


```python
print(train_list)

```

    [['Have been going since 2007 and every meal has been awesome!!', 1], ['Special thanks to Dylan T. for the recommendation on what to order :) All yummy for my tummy.', 1], ['TOTAL WASTE OF TIME.', 0], ["The whole experience was underwhelming, and I think we'll just go to Ninja Sushi next time.", 0], ['I know this is not like the other restaurants at all, something is very off here!', 0], ["Probably never coming back, and wouldn't recommend it.", 0], ['Our server was super nice and checked on us many times.', 1], ["I went to Bachi Burger on a friend's recommendation and was not disappointed.", 1], ['2 Thumbs Up!!', 1], ['If there were zero stars I would give it zero stars.', 0], ['The one down note is the ventilation could use some upgrading.', 0], ['I loved the grilled pizza, reminded me of legit Italian pizza.', 1], ['The staff are also very friendly and efficient.', 1], ['Our server was very nice, and even though he looked a little overwhelmed with all of our needs, he stayed professional and friendly until the end.', 1], ['The meat was pretty dry, I had the sliced brisket and pulled pork.', 0], ['The worst was the salmon sashimi.', 0], ['The cow tongue and cheek tacos are amazing.', 1], ['Awesome service and food.', 1], ['If you want to wait for mediocre food and downright terrible service, then this is the place for you.', 0], ['My husband said she was very rude... did not even apologize for the bad food or anything.', 0], ['We got the food and apparently they have never heard of salt and the batter on the fish was chewy.', 0], ['Tried to go here for lunch and it was a madhouse.', 0], ['So absolutley fantastic.', 1], ["Point your finger at any item on the menu, order it and you won't be disappointed.", 1], ['The only downside is the service.', 0], ["In fact I'm going to round up to 4 stars, just because she was so awesome.", 1], ['The Greek dressing was very creamy and flavorful.', 1], ["(It wasn't busy either) Also, the building was FREEZING cold.", 0], ['The portion was huge!', 1], ['Great pork sandwich.', 1], ['Classy/warm atmosphere, fun and fresh appetizers, succulent steaks (Baseball steak!!!!!', 1], ['This place is disgusting!', 0], ['Food arrived quickly!', 1], ['Interesting decor.', 1], ['The only reason to eat here would be to fill up before a night of binge drinking just to get some carbs in your stomach.', 0], ['Also, I feel like the chips are bought, not made in house.', 0], ['Terrible service!', 0], ['The potatoes were like rubber and you could tell they had been made up ahead of time being kept under a warmer.', 0], ["I go to far too many places and I've never seen any restaurant that serves a 1 egg breakfast, especially for $4.00.", 0], ['Bland and flavorless is a good way of describing the barely tepid meat.', 0], ['I got food poisoning here at the buffet.', 0], ['Def coming back to bowl next time', 1], ["I've never been treated so bad.", 0], ['This is my new fav Vegas buffet spot.', 1], ['He came running after us when he realized my husband had left his sunglasses on the table.', 1], ['If she had not rolled the eyes we may have stayed... Not sure if we will go back and try it again.', 0], ["As for the service: I'm a fan, because it's quick and you're being served by some nice folks.", 1], ['Great place to eat, reminds me of the little mom and pop shops in the San Francisco Bay Area.', 1], ['I hope this place sticks around.', 1], ["It'll be a regular stop on my trips to Phoenix!", 1], ['This was my first crawfish experience, and it was delicious!', 1], ['A great way to finish a great.', 1], ['The kids play area is NASTY!', 0], ["I just don't know how this place managed to served the blandest food I have ever eaten when they are preparing Indian cuisine.", 0], ['The cashew cream sauce was bland and the vegetables were undercooked.', 0], ['Host staff were, for lack of a better word, BITCHES!', 0], ['The dining space is tiny, but elegantly decorated and comfortable.', 1], ['We watched our waiter pay a lot more attention to other tables and ignore us.', 0], ['The inside is really quite nice and very clean.', 1], ['The best place in Vegas for breakfast (just check out a Sat, or Sun.', 1], ['Generous portions and great taste.', 1], ['The only thing I did like was the prime rib and dessert section.', 1], ['Anyways, The food was definitely not filling at all, and for the price you pay you should expect more.', 0], ['They brought a fresh batch of fries and I was thinking yay something warm but no!', 0], ['The food was delicious, our bartender was attentive and personable AND we got a great deal!', 1], ['A greasy, unhealthy meal.', 0], ['The steaks are all well trimmed and also perfectly cooked.', 1], ['Google mediocre and I imagine Smashburger will pop up.', 0], ['Great food.', 1], ['Everything was good and tasty!', 1], ['Also there are combos like a burger, fries, and beer for 23 which is a decent deal.', 1], ["The building itself seems pretty neat; the bathroom is pretty trippy, but I wouldn't eat here again.", 0], ['Be sure to order dessert, even if you need to pack it to-go - the tiramisu and cannoli are both to die for.', 1], ['We waited for thirty minutes to be seated (although there were 8 vacant tables and we were the only folks waiting).', 0], ['Worst Thai ever.', 0], ['If you want healthy authentic or ethic food, try this place.', 1], ['I will not return.', 0], ['I could care less... The interior is just beautiful.', 1], ['The problem I have is that they charge $11.99 for a sandwich that is no bigger than a Subway sub (which offers better and more amount of vegetables).', 0], ["this is the worst sushi i have ever eat besides Costco's.", 0], ['Good prices.', 1], ["After all the rave reviews I couldn't wait to eat here......what a disappointment!", 0], ['An extensive menu provides lots of options for breakfast.', 1], ['Our server was very nice and attentive as were the other serving staff.', 1], ['The turkey and roast beef were bland.', 0], ['Service was fantastic.', 1], ['The jalapeno bacon is soooo good.', 1], ["Hard to judge whether these sides were good because we were grossed out by the melted styrofoam and didn't want to eat it for fear of getting sick.", 0], ['The croutons also taste homemade which is an extra plus.', 1], ['Everything was perfect the night we were in.', 1], ['The food came out at a good pace.', 1], ['This is a really fantastic Thai restaurant which is definitely worth a visit.', 1], ['Pricing is a bit of a concern at Mellow Mushroom.', 0], ['Please stay away from the shrimp stir fried noodles.', 0], ['Their frozen margaritas are WAY too sugary for my taste.', 0], ['The chicken wings contained the driest chicken meat I have ever eaten.', 0], ['Overall, I was very disappointed with the quality of food at Bouchon.', 0], ['Reasonably priced also!', 1], ["Now the burgers aren't as good, the pizza which used to be amazing is doughy and flavorless.", 0], ['Crust is not good.', 0], ['Then our food came out, disappointment ensued.', 0], ["It's worth driving up from Tucson!", 1], ['Both great!', 1], ['I was shocked because no signs indicate cash only.', 0], ['Overpriced for what you are getting.', 0], ["The goat taco didn't skimp on the meat and wow what FLAVOR!", 1], ['Not a single employee came out to see if we were OK or even needed a water refill once they finally served us our food.', 0], ["I've never been more insulted or felt disrespected.", 0], ['Those burgers were amazing.', 1], ['It was pretty gross!', 0], ['It sure does beat the nachos at the movies but I would expect a little bit more coming from a restaurant.', 0], ['We also ordered the spinach and avocado salad; the ingredients were sad and the dressing literally had zero taste.', 0], ['Waitress was a little slow in service.', 0], ['We had 7 at our table and the service was pretty fast.', 1], ["I don't have to be an accountant to know I'm getting screwed!", 0], ["The food wasn't good.", 0], ["Coming here is like experiencing an underwhelming relationship where both parties can't wait for the other person to ask to break up.", 0], ['I will continue to come here on ladies night andddd date night ... highly recommend this place to anyone who is in the area (;', 1], ['The waitress and manager are so friendly.', 1], ['My gyro was basically lettuce only.', 0], ['She was quite disappointed although some blame needs to be placed at her door.', 0], ['A FLY was in my apple juice.. A FLY!!!!!!!!', 0], ['The service was meh.', 0], ['The food sucked, which we expected but it sucked more than we could have imagined.', 0], ['Probably not in a hurry to go back.', 0], ['For service, I give them no stars.', 0], ['Food was average at best.', 0], ['The fries were great too.', 1], ['I\'m so happy to be here!!!"', 1], ['The flair bartenders are absolutely amazing!', 1], ["Paying $7.85 for a hot dog and fries that looks like it came out of a kid's meal at the Wienerschnitzel is not my idea of a good meal.", 0], ['Waiter was a jerk.', 0], ['Thoroughly disappointed!', 0], ['Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.', 1], ['I would definitely recommend the wings as well as the pizza.', 1], ['This place is pretty good, nice little vibe in the restaurant.', 1], ["I started this review with two stars, but I'm editing it to give it only one.", 0], ['It is worth the drive.', 1], ['I will not be eating there again.', 0], ['Service was very prompt.', 1], ['Waitress was good though!', 1], ['However, my recent experience at this particular location was not so good.', 0], ['Great Pizza and Salads!', 1], ['The chipolte ranch dipping sause was tasteless, seemed thin and watered down with no heat.', 0], ['I was so insulted.', 0], ['My breakfast was perpared great, with a beautiful presentation of 3 giant slices of Toast, lightly dusted with powdered sugar.', 1], ['My friend did not like his Bloody Mary.', 0], ['Give it a try, you will be happy you did.', 1], ["All in all, I can assure you I'll be back.", 1], ['I had high hopes for this place since the burgers are cooked over a charcoal grill, but unfortunately the taste fell flat, way flat.', 0], ['Nice ambiance.', 1], ["The place was fairly clean but the food simply wasn't worth it.", 0], ['Your servers suck, wait, correction, our server Heimer sucked.', 0], ["We won't be going back anytime soon!", 0], ['As for the "mains," also uninspired.', 0], ["I promise they won't disappoint.", 1], ['And then tragedy struck.', 0], ['There is so much good food in Vegas that I feel cheated for wasting an eating opportunity by going to Rice and Company.', 0], ['Awesome selection of beer.', 1], ['Never again will I be dining at this place!', 0], ["I find wasting food to be despicable, but this just wasn't food.", 0], ['Mediocre food.', 0], ['Ample portions and good prices.', 1], ['I kept looking at the time and it had soon become 35 minutes, yet still no food.', 0], ['For a self proclaimed coffee cafe, I was wildly disappointed.', 0], ['Overall I was not impressed and would not go back.', 0], ['Thus far, have only visited twice and the food was absolutely delicious each time.', 1], ['The menu is always changing, food quality is going down & service is extremely slow.', 0], ['I immediately said I wanted to talk to the manager but I did not want to talk to the guy who was doing shots of fireball behind the bar.', 0], ['The bartender was also nice.', 1], ['We made the drive all the way from North Scottsdale... and I was not one bit disappointed!', 1], ["You won't be disappointed.", 1], ['I hate those things as much as cheap quality black olives.', 0], ['I had a seriously solid breakfast here.', 1], ['Good value, great food, great service.', 1], ['If you stay in Vegas you must get breakfast here at least once.', 1], ['The ambience is wonderful and there is music playing.', 1], ['The grilled chicken was so tender and yellow from the saffron seasoning.', 1], ["Just don't know why they were so slow.", 0], ['I love their fries and their beans.', 1], ['Great food and service, huge portions and they give a military discount.', 1], ['Loved it...friendly servers, great food, wonderful and imaginative menu.', 1], ["Also, the fries are without a doubt the worst fries I've ever had.", 0], ['It was awesome.', 1], ['Do not waste your money here!', 0], ['The patio seating was very comfortable.', 1], ["I won't be back.", 0], ['I left with a stomach ache and felt sick the rest of the day.', 0], ["It's close to my house, it's low-key, non-fancy, affordable prices, good food.", 1], ['I am far from a sushi connoisseur but I can definitely tell the difference between good food and bad food and this was certainly bad food.', 0], ['She ordered a toasted English muffin that came out untoasted.', 0], ["Great Subway, in fact it's so good when you come here every other Subway will not meet your expectations.", 1], ['Some may say this buffet is pricey but I think you get what you pay for and this place you are getting quite a lot!', 1], ['First - the bathrooms at this location were dirty- Seat covers were not replenished & just plain yucky!!!', 0], ['The food was barely lukewarm, so it must have been sitting waiting for the server to bring it out to us.', 0], ['I had a salad with the wings, and some ice cream for dessert and left feeling quite satisfied.', 1], ['I asked multiple times for the wine list and after some time of being ignored I went to the hostess and got one myself.', 0], ['Gave up trying to eat any of the crust (teeth still sore).', 0], ['I work in the hospitality industry in Paradise Valley and have refrained from recommending Cibo any longer.', 0], ['I have been to very few places to eat that under no circumstances would I ever return to, and this tops the list.', 0], ['This place is way too overpriced for mediocre food.', 0], ['At least think to refill my water before I struggle to wave you over for 10 minutes.', 0], ["When I'm on this side of town, this will definitely be a spot I'll hit up again!", 1], ['On the good side, the staff was genuinely pleasant and enthusiastic - a real treat.', 1], ['RUDE & INCONSIDERATE MANAGEMENT.', 0], ['The seafood was fresh and generous in portion.', 1], ['I have eaten here multiple times, and each time the food was delicious.', 1], ['Will not be back.', 0], ['I probably would not go here again.', 0], ['It was a bit too sweet, not really spicy enough, and lacked flavor.', 0], ['Why are these sad little vegetables so overcooked?', 0], ['This place is like Chipotle, but BETTER.', 1], ['Never going back.', 0], ['I had a pretty satifying experience.', 1], ['Damn good steak.', 1], ['Never been to Hard Rock Casino before, WILL NEVER EVER STEP FORWARD IN IT AGAIN!', 0], ['The salad had just the right amount of sauce to not over power the scallop, which was perfectly cooked.', 1], ['The bathrooms are clean and the place itself is well decorated.', 1], ['The Han Nan Chicken was also very tasty.', 1], ['Great food and awesome service!', 1], ["The warm beer didn't help.", 0], ['You get incredibly fresh fish, prepared with care.', 1], ['Both of the egg rolls were fantastic.', 1], ['After waiting an hour and being seated, I was not in the greatest of moods.', 0], ['We loved the biscuits!!!', 1], ['They really want to make your experience a good one.', 1], ['This place is two thumbs up....way up.', 1], ['He deserves 5 stars.', 1], ['We had fantastic service, and were pleased by the atmosphere.', 1], ['The desserts were a bit strange.', 0], ['I paid the bill but did not tip because I felt the server did a terrible job.', 0], ['Service stinks here!', 0], ['Went in for happy hour, great list of wines.', 1], ['I had to wait over 30 minutes to get my drink and longer to get 2 arepas.', 0], ['I believe that this place is a great stop for those with a huge belly and hankering for sushi.', 1], ['A fantastic neighborhood gem !!!', 1], ['Perfect for someone (me) who only likes beer ice cold, or in this case, even colder.', 1], ['Very very fun chef.', 1], ['The ambiance was incredible.', 1], ['I love this place.', 1], ["I'd say that would be the hardest decision... Honestly, all of M's dishes taste how they are supposed to taste (amazing).", 1], ['Spend your money elsewhere.', 0], ['Dessert: Panna Cotta was amazing.', 1], ['I LOVED their mussels cooked in this wine reduction, the duck was tender, and their potato dishes were delicious.', 1], ['Bad day or not, I have a very low tolerance for rude customer service people, it is your job to be nice and polite, wash dishes otherwise!!', 0], ["This place is a jewel in Las Vegas, and exactly what I've been hoping to find in nearly ten years living here.", 1], ['Great atmosphere, friendly and fast service.', 1], ['We walked away stuffed and happy about our first Vegas buffet experience.', 1], ['Disappointing experience.', 0], ['I will never go back to this place and will never ever recommended this place to anyone!', 0], ['He was terrible!', 0], ['I have been in more than a few bars in Vegas, and do not ever recall being charged for tap water.', 0], ["I'm probably one of the few people to ever go to Ians and not like it.", 0], ['The first time I ever came here I had an amazing experience, I still tell people how awesome the duck was.', 1], ['Their daily specials are always a hit with my group.', 1], ['The staff is great, the food is delish, and they have an incredible beer selection.', 1], ['The service was a little slow , considering that were served by 3 people servers so the food was coming in a slow pace.', 0], ['I had about two bites and refused to eat anymore.', 0], ['Any grandmother can make a roasted chicken better than this one.', 0], ['The Buffet at Bellagio was far from what I anticipated.', 0], ['This wonderful experience made this place a must-stop whenever we are in town again.', 1], ['Very disappointing!!!', 0], ['I love that they put their food in nice plastic containers as opposed to cramming it in little paper takeout boxes.', 1], ['We sat another ten minutes and finally gave up and left.', 0], ['Hot dishes are not hot, cold dishes are close to room temp.I watched staff prepare food with BARE HANDS, no gloves.Everything is deep fried in oil.', 0], ['Always a great time at Dos Gringos!', 1], ['- the food is rich so order accordingly.', 1], ['The best place to go for a tasty bowl of Pho!', 1], ["They have a plethora of salads and sandwiches, and everything I've tried gets my seal of approval.", 1], ['This is a disgrace.', 0], ['The service here is fair at best.', 0], ['The restaurant is very clean and has a family restaurant feel to it.', 1], ['definitely will come back here again.', 1], ["We thought you'd have to venture further away to get good sushi, but this place really hit the spot that night.", 1], ['The food was very good and I enjoyed every mouthful, an enjoyable relaxed venue for couples small family groups etc.', 1], ["My fella got the huevos rancheros and they didn't look too appealing.", 0], ['the potatoes were great and so was the biscuit.', 1], ["If it were possible to give them zero stars, they'd have it.", 0], ['If that bug never showed up I would have given a 4 for sure, but on the other side of the wall where this bug was climbing was the kitchen.', 0], ['No one at the table thought the food was above average or worth the wait that we had for it.', 0], ['We were promptly greeted and seated.', 1], ['Good food , good service .', 1], ['Spend your money and time some place else.', 0], ['Your staff spends more time talking to themselves than me.', 0], ["It kept getting worse and worse so now I'm officially done.", 0], ['The bus boy on the other hand was so rude.', 0], ["It's too bad the food is so damn generic.", 0], ['Everything was gross.', 0], ['Phenomenal food, service and ambiance.', 1], ['I consider this theft.', 0], ['My drink was never empty and he made some really great menu suggestions.', 1], ['We got sitting fairly fast, but, ended up waiting 40 minutes just to place our order, another 30 minutes before the food arrived.', 0], ["I personally love the hummus, pita, baklava, falafels and Baba Ganoush (it's amazing what they do with eggplant!).", 1], ["I've had better bagels from the grocery store.", 0], ['- They never brought a salad we asked for.', 0], ['CONCLUSION: Very filling meals.', 1], ['It was so bad, I had lost the heart to finish it.', 0], ['They were excellent.', 1], ['Their chow mein is so good!', 1], ['Prices are very reasonable, flavors are spot on, the sauce is home made, and the slaw is not drenched in mayo.', 1], ["The ambiance isn't much better.", 0], ['The nachos are a MUST HAVE!', 1], ['I really do recommend this place, you can go wrong with this donut place!', 1], ['But I definitely would not eat here again.', 0], ['They have a good selection of food including a massive meatloaf sandwich, a crispy chicken wrap, a delish tuna melt and some tasty burgers.', 1], ['The atmosphere was great with a lovely duo of violinists playing songs we requested.', 1], ['My ribeye steak was cooked perfectly and had great mesquite flavor.', 1], ['The RI style calamari was a joke.', 0], ["The food is about on par with Denny's, which is to say, not good at all.", 0], ["I don't think I'll be running back to Carly's anytime soon for food.", 0], ['Cute, quaint, simple, honest.', 1], ['I want to first say our server was great and we had perfect service.', 1], ['Their steaks are 100% recommended!', 1], ['Great steak, great sides, great wine, amazing desserts.', 1], ['2 times - Very Bad Customer Service !', 0], ['It was absolutely amazing.', 1], ["Once you get inside you'll be impressed with the place.", 1], ['Worst service to boot, but that is the least of their worries.', 0], ["If you're not familiar, check it out.", 1], ["The servers are not pleasant to deal with and they don't always honor Pizza Hut coupons.", 0], ['The vegetables are so fresh and the sauce feels like authentic Thai.', 1], ['Not good for the money.', 0], ['I liked the patio and the service was outstanding.', 1], ['Always a pleasure dealing with him.', 1], ['We are so glad we found this place.', 1], ['Total letdown, I would much rather just go to the Camelback Flower Shop and Cartel Coffee.', 0], ['Just spicy enough.. Perfect actually.', 1], ["Maybe it's just their Vegetarian fare, but I've been twice and I thought it was average at best.", 0], ['This place is amazing!', 1], ['Things that went wrong: - They burned the saganaki.', 0], ['Went for lunch - service was slow.', 0], ['Good Service-check!', 1], ['I was proven dead wrong by this sushi bar, not only because the quality is great, but the service is fast and the food, impeccable.', 1], ['These were so good we ordered them twice.', 1], ['I will be back many times soon.', 1], ['I was mortified.', 0], ['I waited and waited.', 0], ['The cashier had no care what so ever on what I had to say it still ended up being wayyy overpriced.', 0], ['This place should honestly be blown up.', 0], ['This place has a lot of promise but fails to deliver.', 0], ['There is really nothing for me at postinos, hope your experience is better', 0], ['WAAAAAAyyyyyyyyyy over rated is all I am saying.', 0], ['Server did a great job handling our large rowdy table.', 1], ['I took back my money and got outta there.', 0], ['So flavorful and has just the perfect amount of heat.', 1], ['The steak and the shrimp are in my opinion the best entrees at GC.', 1], ['Will be back again!', 1], ['Frozen pucks of disgust, with some of the worst people behind the register.', 0], ["Definitely a turn off for me & i doubt I'll be back unless someone else is buying.", 0], ["Definitely worth venturing off the strip for the pork belly, will return next time I'm in Vegas.", 1], ['The food is good.', 1], ['will definitely be back!', 1], ['Stopped by this place while in Madison for the Ironman, very friendly, kind staff.', 1], ['I don\'t know what the big deal is about this place, but I won\'t be back "ya\'all".', 0], ['It lacked flavor, seemed undercooked, and dry.', 0], ['The chicken was deliciously seasoned and had the perfect fry on the outside and moist chicken on the inside.', 1], ['The descriptions said "yum yum sauce" and another said "eel sauce", yet another said "spicy mayo"...well NONE of the rolls had sauces on them.', 0], ['The fries were not hot, and neither was my burger.', 0], ['I dont think I will be back for a very long time.', 0], ['I ordered Albondigas soup - which was just warm - and tasted like tomato soup with frozen meatballs.', 0], ['Service is quick and even "to go" orders are just like we like it!', 1], ['Would not recommend to others.', 0], ['OMG I felt like I had never eaten Thai food until this dish.', 1], ['They had a toro tartare with a cavier that was extraordinary and I liked the thinly sliced wagyu with white truffle.', 1], ['I had the mac salad and it was pretty bland so I will not be getting that again.', 0], ['did not like at all.', 0], ["This is some seriously good pizza and I'm an expert/connisseur on the topic.", 1], ['Of all the dishes, the salmon was the best, but all were great.', 1], ['Do yourself a favor and stay away from this dish.', 0], ['The only redeeming quality of the restaurant was that it was very inexpensive.', 1], ['I love the owner/chef, his one authentic Japanese cool dude!', 1], ['Cooked to perfection and the service was impeccable.', 1], ['So good I am going to have to review this place twice - once hereas a tribute to the place and once as a tribute to an event held here last night.', 1], ['You cant go wrong with any of the food here.', 1], ['Waitress was sweet and funny.', 1], ["If you are reading this please don't go there.", 0], ['This place was such a nice surprise!', 1], ['The Wife hated her meal (coconut shrimp), and our friends really did not enjoy their meals, either.', 0], ['Not tasty and the texture was just nasty.', 0], ['By this time our side of the restaurant was almost empty so there was no excuse.', 0], ["I can't tell you how disappointed I was.", 0], ['There is nothing privileged about working/eating there.', 0], ['Lordy, the Khao Soi is a dish that is not to be missed for curry lovers!', 1], ["We asked for the bill to leave without eating and they didn't bring that either.", 0], ["I also had to taste my Mom's multi-grain pumpkin pancakes with pecan butter and they were amazing, fluffy, and delicious!", 1], ['Great service and food.', 1], ['I can say that the desserts were yummy.', 1], ['They could serve it with just the vinaigrette and it may make for a better overall dish, but it was still very good.', 1], ['It was not good.', 0], ['The feel of the dining room was more college cooking course than high class dining and the service was slow at best.', 0], ['I guess maybe we went on an off night but it was disgraceful.', 0], ["Then, as if I hadn't wasted enough of my life there, they poured salt in the wound by drawing out the time it took to bring the check.", 0], ['I love the Pho and the spring rolls oh so yummy you have to try.', 1], ['The service was terrible though.', 0], ['Delicious NYC bagels, good selections of cream cheese, real Lox with capers even.', 1], ["Ryan's Bar is definitely one Edinburgh establishment I won't be revisiting.", 0], ["Maybe if they weren't cold they would have been somewhat edible.", 0], ['Favorite place in town for shawarrrrrrma!!!!!!', 1], ['On three different occasions I asked for well done or medium well, and all three times I got the bloodiest piece of meat on my plate.', 0], ['Fantastic service here.', 1], ["I wasn't really impressed with Strip Steak.", 0], ['Service is quick and friendly.', 1], ['Very good, though!', 1], ['Great time - family dinner on a Sunday night.', 1], ['Poor service, the waiter made me feel like I was stupid every time he came to the table.', 0], ['Will never, ever go back.', 0], ['Will not be back!', 0], ['We could not believe how dirty the oysters were!', 0], ['Now the pizza itself was good the peanut sauce was very tasty.', 1], ['The pancake was also really good and pretty large at that.', 1], ['Also were served hot bread and butter, and home made potato chips with bacon bits on top....very original and very good.', 1], ['I loved the bacon wrapped dates.', 1], ["Third, the cheese on my friend's burger was cold.", 0], ['This is a great restaurant at the Mandalay Bay.', 1], ['a drive thru means you do not want to wait around for half an hour for your food, but somehow when we end up going here they make us wait and wait.', 0], ["Won't go back.", 0], ['The pizza tasted old, super chewy in not a good way.', 0], ['Needless to say, we will never be back here again.', 0], ["It wasn't busy at all and now we know why.", 0], ['They dropped more than the ball.', 0], ['Took an hour to get our food only 4 tables in restaurant my food was Luke warm, Our sever was running around like he was totally overwhelmed.', 0], ['The chips that came out were dripping with grease, and mostly not edible.', 0], ['The lighting is just dark enough to set the mood.', 1], ["They will customize your order any way you'd like, my usual is Eggplant with Green Bean stir fry, love it!", 1], ["I'd rather eat airline food, seriously.", 0], ['Everyone is treated equally special.', 1], ['Appetite instantly gone.', 0], ['Our waiter was very attentive, friendly, and informative.', 1], ['It was way over fried.', 0], ["For about 10 minutes, we we're waiting for her salad when we realized that it wasn't coming any time soon.", 0], ['The seasonal fruit was fresh white peach puree.', 1], ['I found this place by accident and I could not be happier.', 1], ['The classic Maine Lobster Roll was fantastic.', 1], ['My side Greek salad with the Greek dressing was so tasty, and the pita and hummus was very refreshing.', 1], ['Seriously flavorful delights, folks.', 1], ['I can take a little bad service but the food sucks.', 0], ['But then they came back cold.', 0], ['The owners are super friendly and the staff is courteous.', 1], ['This place is not worth your time, let alone Vegas.', 0], ['What SHOULD have been a hilarious, yummy Christmas Eve dinner to remember was the biggest fail of the entire trip for us.', 0], ['I came back today since they relocated and still not impressed.', 0], ['The Heart Attack Grill in downtown Vegas is an absolutely flat-lined excuse for a restaurant.', 0], ['It took over 30 min to get their milkshake, which was nothing more than chocolate milk.', 0], ['Their monster chicken fried steak and eggs is my all time favorite.', 1], ['So in a nutshell: 1) The restaraunt smells like a combination of a dirty fish market and a sewer.', 0], ['This was my first and only Vegas buffet and it did not disappoint.', 1], ['This is one of the better buffets that I have been to.', 1], ['Hopefully this bodes for them going out of business and someone who can cook can come in.', 0], ['When my mom and I got home she immediately got sick and she only had a few bites of salad.', 0], ['If you want a sandwich just go to any Firehouse!!!!!', 1], ['We literally sat there for 20 minutes with no one asking to take our order.', 0], ['The deal included 5 tastings and 2 drinks, and Jeff went above and beyond what we expected.', 1], ['It was attached to a gas station, and that is rarely a good sign.', 0], ['Very good food, great atmosphere.1', 1], ["One nice thing was that they added gratuity on the bill since our party was larger than 6 or 8, and they didn't expect more tip than that.", 1], ['We loved the place.', 1], ['The food was great as always, compliments to the chef.', 1], ["The shower area is outside so you can only rinse, not take a full shower, unless you don't mind being nude for everyone to see!", 0], ['We had so much to say about the place before we walked in that he expected it to be amazing, but was quickly disappointed.', 0], ['Unfortunately, it was not good.', 0], ['The food was terrible.', 0], ['Update.....went back for a second time and it was still just as amazing', 1], ["I won't try going back there even if it's empty.", 0], ["We aren't ones to make a scene at restaurants but I just don't get it...definitely lost the love after this one!", 0], ['Great food and great service in a clean and friendly setting.', 1], ['Cant say enough good things about this place.', 1], ['The staff are great, the ambiance is great.', 1], ['Their menu is diverse, and reasonably priced.', 1], ["I wouldn't return.", 0], ["We won't be going back.", 0], ['Food was below average.', 0], ['Best service and food ever, Maria our server was so good and friendly she made our day.', 1], ['It also took her forever to bring us the check when we asked for it.', 0], ['I have watched their prices inflate, portions get smaller and management attitudes grow rapidly!', 0], ["Based on the sub-par service I received and no effort to show their gratitude for my business I won't be going back.", 0], ["Overall, I don't think that I would take my parents to this place again because they made most of the similar complaints that I silently felt too.", 0], ['The potato chip order was sad... I could probably count how many chips were in that box and it was probably around 12.', 0], ["I don't recommend unless your car breaks down in front of it and you are starving.", 0], ['the staff is friendly and the joint is always clean.', 1], ['This was like the final blow!', 0], ['Although I very much liked the look and sound of this place, the actual experience was a bit disappointing.', 0], ['I have never had such bland food which surprised me considering the article we read focused so much on their spices and flavor.', 0], ['Sauce was tasteless.', 0], ['After two I felt disgusting.', 0], ["It shouldn't take 30 min for pancakes and eggs.", 0], ["We started with the tuna sashimi which was brownish in color and obviously wasn't fresh.", 0], ['Nice, spicy and tender.', 1], ["I mean really, how do you get so famous for your fish and chips when it's so terrible!?!", 0], ['the food is not tasty at all, not to say its "real traditional Hunan style".', 0], ["Worst food/service I've had in a while.", 0], ['Similarly, the delivery man did not say a word of apology when our food was 45 minutes late.', 0], ['First time going but I think I will quickly become a regular.', 1], ["I'm not impressed with the concept or the food.", 0], ['The vanilla ice cream was creamy and smooth while the profiterole (choux) pastry was fresh enough.', 1], ['Con: spotty service.', 0], ['After 20 minutes wait, I got a table.', 0], ["Don't do it!!!!", 0], ['Not to mention the combination of pears, almonds and bacon is a big winner!', 1], ['In the summer, you can dine in a charming outdoor patio - so very delightful.', 1], ['The roast beef sandwich tasted really good!', 1], ['Pretty cool I would say.', 1], ["I didn't know pulled pork could be soooo delicious.", 1], ['The service was a bit lacking.', 0], ['It was equally awful.', 0], ["The only good thing was our waiter, he was very helpful and kept the bloddy mary's coming.", 1], ['My sashimi was poor quality being soggy and tasteless.', 0], ['Four stars for the food & the guy in the blue shirt for his great vibe & still letting us in to eat !', 1], ['Best tacos in town by far!!', 1], ['We definately enjoyed ourselves.', 1], ['The service here leaves a lot to be desired.', 0], ['very tough and very short on flavor!', 0], ["If you haven't gone here GO NOW!", 1], ['* Both the Hot & Sour & the Egg Flower Soups were absolutely 5 Stars!', 1], ["I probably won't be coming back here.", 0], ["REAL sushi lovers, let's be honest - Yama is not that good.", 0], ['This place has it!', 1], ["They have horrible attitudes towards customers, and talk down to each one when customers don't enjoy their food.", 0], ['I love the fact that everything on their menu is worth it.', 1], ['The sweet potato tots were good but the onion rings were perfection or as close as I have had.', 1], ["Bland... Not a liking this place for a number of reasons and I don't want to waste time on bad reviewing.. I'll leave it at that...", 0], ['Will go back next trip out.', 1], ['I dressed up to be treated so rudely!', 0], ['it was a drive to get there.', 0], ['After one bite, I was hooked.', 1], ['Sorry, I will not be getting food from here anytime soon :(', 0], ['The ambiance here did not feel like a buffet setting, but more of a douchey indoor garden for tea and biscuits.', 0], ['Restaurant is always full but never a wait.', 1], ['No complaints!', 1], ['Much better than the other AYCE sushi place I went to in Vegas.', 1], ['He also came back to check on us regularly, excellent service.', 1], ["I'm super pissd.", 0], ['The place was not clean and the food oh so stale!', 0], ["Hawaiian Breeze, Mango Magic, and Pineapple Delight are the smoothies that I've tried so far and they're all good.", 1], ["I've had better, not only from dedicated boba tea spots, but even from Jenni Pho.", 0], ['In an interesting part of town, this place is amazing.', 1], ['I LOVED it!', 1], ['Service was excellent and prices are pretty reasonable considering this is Vegas and located inside the Crystals shopping mall by Aria.', 1], ['A good time!', 1], ['The service was poor and thats being nice.', 0], ['The scallop dish is quite appalling for value as well.', 0], ['Waited 2 hours & never got either of our pizzas as many other around us who came in later did!', 0], ['This is an unbelievable BARGAIN!', 1], ["Don't bother coming here.", 0], ['The service was outshining & I definitely recommend the Halibut.', 1], ['Kind of hard to mess up a steak but they did.', 0], ['All I have to say is the food was amazing!!!', 1], ['Very friendly staff.', 1], ["I'm not really sure how Joey's was voted best hot dog in the Valley by readers of Phoenix Magazine.", 0], ["By this point, my friends and I had basically figured out this place was a joke and didn't mind making it publicly and loudly known.", 0], ["I can't wait to go back.", 1], ['I *heart* this place.', 1], ['No allergy warnings on the menu, and the waitress had absolutely no clue as to which meals did or did not contain peanuts.', 0], ['From what my dinner companions told me...everything was very fresh with nice texture and taste.', 1], ['I did not expect this to be so good!', 1], ['Not a weekly haunt, but definitely a place to come back to every once in a while.', 1], ['The food was very good.', 1], ["I probably won't be back, to be honest.", 0], ['All of the tapas dishes were delicious!', 1], ['This greedy corporation will NEVER see another dime from me!', 0], ['Despite how hard I rate businesses, its actually rare for me to give a 1 star.', 0], ['Never had anything to complain about here.', 1], ["Needless to say, I won't be going back anytime soon.", 0], ['The Veggitarian platter is out of this world!', 1], ["Don't waste your time here.", 0], ["I'm not eating here!", 0], ['Very, very sad.', 0], ['What happened next was pretty....off putting.', 0], ['As always the evening was wonderful and the food delicious!', 1], ['The manager was the worst.', 0], ["If someone orders two tacos don't' you think it may be part of customer service to ask if it is combo or ala cart?", 0], ['Del Taco is pretty nasty and should be avoided if possible.', 0], ['Penne vodka excellent!', 1], ["Nicest Chinese restaurant I've been in a while.", 1], ['I was seated immediately.', 1], ['What a mistake.', 0], ['The chicken dishes are OK, the beef is like shoe leather.', 0], ["I ordered the Voodoo pasta and it was the first time I'd had really excellent pasta since going gluten free several years ago.", 1], ['A lady at the table next to us found a live green caterpillar In her salad.', 0], ['When my order arrived, one of the gyros was missing.', 0], ['However, there was so much garlic in the fondue, it was barely edible.', 0], ['Come hungry, leave happy and stuffed!', 1], ['The price is reasonable and the service is great.', 1], ['Lobster Bisque, Bussell Sprouts, Risotto, Filet ALL needed salt and pepper..and of course there is none at the tables.', 0], ['They have great dinners.', 1], ['Service was fine and the waitress was friendly.', 1], ['They were golden-crispy and delicious.', 1], ["Once your food arrives it's meh.", 0], ['Talk about great customer service of course we will be back.', 1], ['This place is hands-down one of the best places to eat in the Phoenix metro area.', 1], ['The management is rude.', 0], ['My wife had the Lobster Bisque soup which was lukewarm.', 0], ['The black eyed peas and sweet potatoes... UNREAL!', 1], ['The pan cakes everyone are raving about taste like a sugary disaster tailored to the palate of a six year old.', 0], ['Service was slow and not attentive.', 0], ['Check it out.', 1], ['This hole in the wall has great Mexican street tacos, and friendly staff.', 1], ["So don't go there if you are looking for good food...", 0], ['Tasted like dirt.', 0], ['5 stars for the brick oven bread app!', 1], ['Im in AZ all the time and now have my new spot.', 1], ["How can you call yourself a steakhouse if you can't properly cook a steak, I don't understand!", 0], ['!....THE OWNERS REALLY REALLY need to quit being soooooo cheap let them wrap my freaking sandwich in two papers not one!', 0], ['Today was my first taste of a Buldogis Gourmet Hot Dog and I have to tell you it was more than I ever thought possible.', 1], ['I seriously cannot believe that the owner has so many unexperienced employees that all are running around like chickens with their heads cut off.', 0], ['My friend loved the salmon tartar.', 1], ['I give it 2 thumbs down', 0], ['I love this place.', 1], ['But now I was completely grossed out.', 0], ['Hell no will I go back', 0], ['Not good by any stretch of the imagination.', 0], ['This place deserves one star and 90% has to do with the food.', 0], ['As a sushi lover avoid this place by all means.', 0], ['Must have been an off night at this place.', 0], ['My fiancÃ© and I came in the middle of the day and we were greeted and seated right away.', 1], ['Very good lunch spot.', 1], ['Over rated.', 0], ['It was just not a fun experience.', 1], ['Everything was fresh and delicious!', 1], ['The pizza selections are good.', 1], ['Very poor service.', 0], ['I never come again.', 0], ['This place receives stars for their APPETIZERS!!!', 1], ['Nargile - I think you are great.', 1], ['The selection on the menu was great and so were the prices.', 1], ['Overall, a great experience.', 1], ["I will come back here every time I'm in Vegas.", 1], ["I've had better atmosphere.", 0], ['Extremely Tasty!', 1], ['I love the decor with the Chinese calligraphy wall paper.', 1], ["It really is impressive that the place hasn't closed down.", 0], ['The food was excellent and service was very good.', 1], ['Seafood was limited to boiled shrimp and crab legs but the crab legs definitely did not taste fresh.', 0], ['Food was really boring.', 0], ['This is the place where I first had pho and it was amazing!!', 1], ["too bad cause I know it's family owned, I really wanted to like this place.", 0], ['Bacon is hella salty.', 1], ['-My order was not correct.', 0], ['Their regular toasted bread was equally satisfying with the occasional pats of butter... Mmmm...!', 1], ['DELICIOUS!!', 1], ['This is was due to the fact that it took 20 minutes to be acknowledged, then another 35 minutes to get our food...and they kept forgetting things.', 0], ['Insults, profound deuchebaggery, and had to go outside for a smoke break while serving just to solidify it.', 0], ['Pretty good beer selection too.', 1], ['The food is very good for your typical bar food.', 1], ['And service was super friendly.', 1], ['Hands down my favorite Italian restaurant!', 1], ['My first visit to Hiro was a delight!', 1], ['The selection of food was not the best.', 0], ['On a positive note, our server was very attentive and provided great service.', 1], ['Just as good as when I had it more than a year ago!', 1], ['The burger had absolutely no flavor - the meat itself was totally bland, the burger was overcooked and there was no charcoal flavor.', 0], ['The service was great, even the manager came and helped with our table.', 1], ['Our server was fantastic and when he found out the wife loves roasted garlic and bone marrow, he added extra to our meal and another marrow to go!', 1], ['A great touch.', 1], ['They also now serve Indian naan bread with hummus and some spicy pine nut sauce that was out of this world.', 1], ['By far the BEST cheesecurds we have ever had!', 1], ['There is not a deal good enough that would drag me into that establishment again.', 0], ['Sooooo good!!', 1], ['There was hardly any meat.', 0], ['We recently witnessed her poor quality of management towards other guests as well.', 0], ['The live music on Fridays totally blows.', 0], ['The poor batter to meat ratio made the chicken tenders very unsatisfying.', 0], ['The service was not up to par, either.', 0], ['That said, our mouths and bellies were still quite pleased.', 1], ["As much as I'd like to go back, I can't get passed the atrocious service and will never return.", 0], ["I could barely stomach the meal, but didn't complain because it was a business lunch.", 0], ['It was either too cold, not enough flavor or just bad.', 0], ["We've have gotten a much better service from the pizza place next door than the services we received from this restaurant.", 0], ['Ordered a double cheeseburger & got a single patty that was falling apart (picture uploaded) Yeah, still sucks.', 0], ['like the other reviewer said "you couldn\'t pay me to eat at this place again."', 0], ['I got to enjoy the seafood salad, with a fabulous vinegrette.', 1], ['Not much seafood and like 5 strings of pasta at the bottom.', 0], ['Unfortunately, it only set us up for disapppointment with our entrees.', 0], ["I live in the neighborhood so I am disappointed I won't be back here, because it is a convenient location.", 0], ['The service was terrible, food was mediocre.', 0], ['The staff was very attentive.', 1], ['At first glance it is a lovely bakery cafe - nice ambiance, clean, friendly staff.', 1], ['We enjoy their pizza and brunch.', 1], ['The ripped banana was not only ripped, but petrified and tasteless.', 0], ['This place is not quality sushi, it is not a quality restaurant.', 0], ['The staff is always super friendly and helpful, which is especially cool when you bring two small boys and a baby!', 1], ['The chicken I got was definitely reheated and was only ok, the wedges were cold and soggy.', 0], ['I had the chicken Pho and it tasted very bland.', 0], ['As for the service, I thought it was good.', 1], ["Main thing I didn't enjoy is that the crowd is of older crowd, around mid 30s and up.", 0], ["No, I'm going to eat the potato that I found some strangers hair in it.", 0], ['An excellent new restaurant by an experienced Frenchman.', 1], ['We waited an hour for what was a breakfast I could have done 100 times better at home.', 0], ['We had a group of 70+ when we claimed we would only have 40 and they handled us beautifully.', 1], ["We've tried to like this place but after 10+ times I think we're done with them.", 0], ["I'm not sure how long we stood there but it was long enough for me to begin to feel awkwardly out of place.", 0], ['-Drinks took close to 30 minutes to come out at one point.', 0], ['Oh this is such a thing of beauty, this restaurant.', 1], ['The fried rice was dry as well.', 0], ['My boyfriend tried the Mediterranean Chicken Salad and fell in love.', 1], ['Left very frustrated.', 0], ['Service is also cute.', 1], ['Omelets are to die for!', 1], ['In summary, this was a largely disappointing dining experience.', 0], ['That just SCREAMS "LEGIT" in my book...somethat\'s also pretty rare here in Vegas.', 1], ['My boyfriend and I came here for the first time on a recent trip to Vegas and could not have been more pleased with the quality of food and service.', 1], ['The Jamaican mojitos are delicious.', 1], ['The chips and salsa were really good, the salsa was very fresh.', 1], ["I don't know what kind it is but they have the best iced tea.", 1], ['I tried the Cape Cod ravoli, chicken,with cranberry...mmmm!', 1], ['first time there and might just be the last.', 0], ['I would recommend saving room for this!', 1], ["My girlfriend's veal was very bad.", 0], ['Lastly, the mozzarella sticks, they were the best thing we ordered.', 1], ['- Really, really good rice, all the time.', 1], ["I won't be back.", 0], ['Definitely not worth the $3 I paid.', 0], ['One of the few places in Phoenix that I would definately go back to again .', 1], ['Great brunch spot.', 1], ['The refried beans that came with my meal were dried out and crusty and the food was bland.', 0], ["Furthermore, you can't even find hours of operation on the website!", 0], ['I just wanted to leave.', 0], ['Join the club and get awesome offers via email.', 1], ['The folks at Otto always make us feel so welcome and special.', 1], ['What did bother me, was the slow service.', 0], ['The last 3 times I had lunch here has been bad.', 0], ['Total brunch fail.', 0], ['The steak was amazing...rge fillet relleno was the best seafood plate i have ever had!', 1], ['Food was good, service was good, Prices were good.', 1], ['I could eat their bruschetta all day it is devine.', 1], ['They have a really nice atmosphere.', 1], ['The restaurant atmosphere was exquisite.', 1], ["This was my first time and I can't wait until the next.", 1], ['It was extremely "crumby" and pretty tasteless.', 0], ['Last night was my second time dining here and I was so happy I decided to go back!', 1], ['Anyway, this FS restaurant has a wonderful breakfast/lunch.', 1], ['Service was exceptional and food was a good as all the reviews.', 1], ['High-quality chicken on the chicken Caesar salad.', 1], ['Try them in the airport to experience some tasty food and speedy, friendly service.', 1], ["And considering the two of us left there very full and happy for about $20, you just can't go wrong.", 1], ['The block was amazing.', 1], ["You can't beat that.", 1], ['I got home to see the driest damn wings ever!', 0], ['Overall, I like there food and the service.', 1], ['Just had lunch here and had a great experience.', 1], ['very slow at seating even with reservation.', 0], ['I would avoid this place if you are staying in the Mirage.', 0], ['for 40 bucks a head, i really expect better food.', 0], ['We ordered the duck rare and it was pink and tender on the inside with a nice char on the outside.', 1], ['Shrimp- When I unwrapped it (I live only 1/2 a mile from Brushfire) it was literally ice cold.', 0], ['I checked out this place a couple years ago and was not impressed.', 0], ['This place is great!!!!!!!!!!!!!!', 1], ['The real disappointment was our waiter.', 0], ['Waited and waited and waited.', 0], ['We waited for forty five minutes in vain.', 0], ['This place is horrible and way overpriced.', 0], ['Fantastic food!', 1], ["Friend's pasta -- also bad, he barely touched it.", 0], ["The selection was probably the worst I've seen in Vegas.....there was none.", 0], ['the presentation of the food was awful.', 0], ['How awesome is that.', 1], ['Wow... Loved this place.', 1], ['I had strawberry tea, which was good.', 1], ["Won't ever go here again.", 0], ['This is one of the best bars with food in Vegas.', 1], ['The sangria was about half of a glass wine full and was $12, ridiculous.', 0], ['On the up side, their cafe serves really good food.', 1], ['It is PERFECT for a sit-down family meal or get together with a few friends.', 1], ['you can watch them preparing the delicious food!)', 1], ['The servers went back and forth several times, not even so much as an "Are you being helped?"', 0], ['So they performed.', 1], ['My brother in law who works at the mall ate here same day, and guess what he was sick all night too.', 0], ['This really is how Vegas fine dining used to be, right down to the menus handed to the ladies that have no prices listed.', 1], ['The cashier was friendly and even brought the food out to me.', 1], ['The server was very negligent of our needs and made us feel very unwelcome... I would not suggest this place!', 0], ['After the disappointing dinner we went elsewhere for dessert.', 0], ["Level 5 spicy was perfect, where spice didn't over-whelm the soup.", 1], ["I recently tried Caballero's and I have been back every week since!", 1], ["I'll take my business dinner dollars elsewhere.", 0], ['Food quality has been horrible.', 0], ['Which are small and not worth the price.', 0], ["At least 40min passed in between us ordering and the food arriving, and it wasn't that busy.", 0], ['The food was terrible.', 0], ['Some highlights : Great quality nigiri here!', 1], ["Owner's are really great people.!", 1], ['Eclectic selection.', 1], ["Couldn't ask for a more satisfying meal.", 1], ['For sushi on the Strip, this is the place to go.', 1], ['What a mistake that was!', 0], ['Wonderful lil tapas and the ambience made me feel all warm and fuzzy inside.', 1], ['The sweet potato fries were very good and seasoned well.', 1], ['The owner used to work at Nobu, so this place is really similar for half the price.', 1], ['But the service was beyond bad.', 0], ['It was packed!!', 0], ['All the bread is made in-house!', 1], ["If the food isn't bad enough for you, then enjoy dealing with the world's worst/annoying drunk people.", 0], ["I guess I should have known that this place would suck, because it is inside of the Excalibur, but I didn't use my common sense.", 0], ['Very bad Experience!', 0], ['Eew... This location needs a complete overhaul.', 0], ['So we went to Tigerlilly and had a fantastic afternoon!', 1], ['I miss it and wish they had one in Philadelphia!', 1], ["The only thing I wasn't too crazy about was their guacamole as I don't like it purÃ©ed.", 0], ['Boy was that sucker dry!!.', 0], ['OMG, the food was delicioso!', 1], ["It's NOT hard to make a decent hamburger.", 0], ['All in all, Ha Long Bay was a bit of a flop.', 0]]
    

# Building a vocabulary as list.


```python
for i in range(len(train_set)):
    if(train_list[i][1] == 1):
      for word in train_list[i][0].split(" "):
          positive_count[word.lower()] += 1
          total_count[word.lower()] += 1
    else:
        for word in train_list[i][0].split(" "):
            negative_count[word.lower()] += 1
            total_count[word.lower()] += 1
```


```python
total_count.most_common()
```




    [('the', 474),
     ('and', 294),
     ('was', 248),
     ('i', 230),
     ('a', 187),
     ('to', 177),
     ('is', 135),
     ('this', 111),
     ('of', 97),
     ('not', 95),
     ('for', 94),
     ('it', 89),
     ('in', 82),
     ('food', 77),
     ('we', 67),
     ('place', 60),
     ('my', 58),
     ('very', 57),
     ('be', 56),
     ('so', 54),
     ('that', 53),
     ('with', 52),
     ('had', 52),
     ('service', 49),
     ('good', 49),
     ('were', 48),
     ('they', 48),
     ('have', 47),
     ('at', 47),
     ('are', 47),
     ('you', 47),
     ('great', 46),
     ('but', 46),
     ('on', 43),
     ('our', 35),
     ('like', 32),
     ('will', 32),
     ('just', 31),
     ('as', 30),
     ('here', 29),
     ('go', 28),
     ('back', 28),
     ('all', 27),
     ('time', 27),
     ('really', 27),
     ('their', 27),
     ('if', 24),
     ('never', 23),
     ('would', 22),
     ('an', 22),
     ('there', 21),
     ('only', 21),
     ('been', 20),
     ('what', 20),
     ("don't", 20),
     ('one', 19),
     ('your', 19),
     ('by', 19),
     ('out', 19),
     ('no', 19),
     ('get', 18),
     ('-', 18),
     ('from', 18),
     ('also', 17),
     ('food.', 17),
     ('did', 17),
     ("won't", 17),
     ("i'm", 17),
     ('came', 17),
     ('when', 17),
     ('good.', 17),
     ('going', 16),
     ('some', 16),
     ('he', 16),
     ('or', 16),
     ('got', 16),
     ('up', 16),
     ('ever', 16),
     ('more', 16),
     ('definitely', 16),
     ('which', 16),
     ('chicken', 16),
     ('us', 15),
     ('pretty', 15),
     ('eat', 15),
     ('first', 15),
     ('nice', 14),
     ("i've", 14),
     ('than', 14),
     ('about', 14),
     ('it.', 13),
     ('could', 13),
     ('friendly', 13),
     ('even', 13),
     ('service.', 13),
     ('made', 13),
     ('restaurant', 13),
     ('how', 13),
     ('back.', 13),
     ('much', 13),
     ('server', 12),
     ('again.', 12),
     ("it's", 12),
     ('better', 12),
     ('best', 12),
     ('minutes', 12),
     ('place.', 12),
     ("didn't", 12),
     ('quality', 12),
     ('can', 12),
     ('has', 11),
     ('other', 11),
     ('went', 11),
     ('me', 11),
     ('staff', 11),
     ('bad', 11),
     ('any', 11),
     ('being', 11),
     ('&', 11),
     ('love', 11),
     ('think', 10),
     ('loved', 10),
     ('because', 10),
     ("wasn't", 10),
     ('feel', 10),
     ('vegas', 10),
     ('after', 10),
     ('worth', 10),
     ('still', 10),
     ('always', 10),
     ('say', 10),
     ('know', 9),
     ('probably', 9),
     ('little', 9),
     ('want', 9),
     ('wait', 9),
     ('fresh', 9),
     ('too', 9),
     ('way', 9),
     ('pizza', 9),
     ("can't", 9),
     ('come', 9),
     ('do', 9),
     ('order', 8),
     ('sushi', 8),
     ('coming', 8),
     ('recommend', 8),
     ('give', 8),
     ('down', 8),
     ('worst', 8),
     ('she', 8),
     ('night', 8),
     ('fries', 8),
     ('food,', 8),
     ('taste', 8),
     ('bit', 8),
     ('two', 8),
     ('over', 8),
     ('here.', 8),
     ('every', 7),
     ('experience', 7),
     ('next', 7),
     ('tried', 7),
     ('sauce', 7),
     ('dining', 7),
     ('thing', 7),
     ('waited', 7),
     ('perfect', 7),
     ('fantastic', 7),
     ('now', 7),
     ('ordered', 7),
     ('slow', 7),
     ('them', 7),
     ('happy', 7),
     ('absolutely', 7),
     ('salad', 7),
     ('times', 7),
     ('experience.', 7),
     ('make', 7),
     ('dishes', 7),
     ('another', 7),
     ('took', 7),
     ('enough', 7),
     ('meal', 6),
     ('off', 6),
     ('super', 6),
     ('many', 6),
     ('2', 6),
     ('meat', 6),
     ('amazing.', 6),
     ('awesome', 6),
     ('service,', 6),
     ('then', 6),
     ('said', 6),
     ('chips', 6),
     ('buffet', 6),
     ('quite', 6),
     ('everything', 6),
     ('menu', 6),
     ('getting', 6),
     ('good,', 6),
     ('once', 6),
     ('felt', 6),
     ('restaurant.', 6),
     ('waitress', 6),
     ('who', 6),
     ('her', 6),
     ('hot', 6),
     ("i'll", 6),
     ('selection', 6),
     ('wonderful', 6),
     ('few', 6),
     ('5', 6),
     ('potato', 6),
     ('vegas.', 6),
     ('enjoy', 6),
     ('poor', 6),
     ('around', 6),
     ('take', 6),
     ('since', 5),
     ('time.', 5),
     ('all,', 5),
     ('here!', 5),
     ('disappointed.', 5),
     ('lunch', 5),
     ('tell', 5),
     ('kept', 5),
     ('far', 5),
     ('barely', 5),
     ('bad.', 5),
     ('left', 5),
     ('table.', 5),
     ('may', 5),
     ('sure', 5),
     ('served', 5),
     ('delicious!', 5),
     ('great.', 5),
     ('inside', 5),
     ('breakfast', 5),
     ('check', 5),
     ('should', 5),
     ('expect', 5),
     ('up.', 5),
     ('beer', 5),
     ('both', 5),
     ('staff.', 5),
     ('bacon', 5),
     ('hard', 5),
     ('amazing', 5),
     ('best.', 5),
     ('great,', 5),
     ('clean', 5),
     ('bring', 5),
     ('asked', 5),
     ('seafood', 5),
     ('delicious.', 5),
     ('spicy', 5),
     ('30', 5),
     ('tasty', 5),
     ('family', 5),
     ('thought', 5),
     ('prices', 5),
     ('steak', 5),
     ('found', 5),
     ('tasted', 5),
     ('dinner', 5),
     ('bread', 5),
     ('excellent', 5),
     ('waste', 4),
     ('burger', 4),
     ('zero', 4),
     ('stars', 4),
     ('mediocre', 4),
     ('terrible', 4),
     ('fantastic.', 4),
     ('fact', 4),
     ('cold.', 4),
     ('before', 4),
     ('places', 4),
     ('bland', 4),
     ('spot.', 4),
     ('running', 4),
     ('his', 4),
     ('try', 4),
     ('cream', 4),
     ('waiter', 4),
     ('pay', 4),
     ('portions', 4),
     ('taste.', 4),
     ('warm', 4),
     ('well', 4),
     ('itself', 4),
     ('sandwich', 4),
     ('away', 4),
     ('fried', 4),
     ('overall,', 4),
     ('disappointed', 4),
     ('where', 4),
     ('manager', 4),
     ('friendly.', 4),
     ('too.', 4),
     ('eating', 4),
     ('cooked', 4),
     ('servers', 4),
     ('anytime', 4),
     ('impressed', 4),
     ('talk', 4),
     ('seriously', 4),
     ('must', 4),
     ('least', 4),
     ('money', 4),
     ('close', 4),
     ('am', 4),
     ('waiting', 4),
     ('ice', 4),
     ('side', 4),
     ('real', 4),
     ('flavor.', 4),
     ('hour', 4),
     ('someone', 4),
     ('ambiance', 4),
     ("i'd", 4),
     ('customer', 4),
     ('people', 4),
     ('considering', 4),
     ('home', 4),
     ('out.', 4),
     ('deal', 4),
     ('there.', 4),
     ('long', 4),
     ('it!', 4),
     ('sweet', 4),
     ('that.', 4),
     ('business', 4),
     ('full', 4),
     ('tasteless.', 4),
     ('well.', 4),
     ('pasta', 4),
     ('live', 4),
     ('recommendation', 3),
     ('yummy', 3),
     ('total', 3),
     ('back,', 3),
     ("wouldn't", 3),
     ("friend's", 3),
     ('thumbs', 3),
     ('stars.', 3),
     ('until', 3),
     ('salmon', 3),
     ('tacos', 3),
     ('salt', 3),
     ('fish', 3),
     ('4', 3),
     ('stars,', 3),
     ('greek', 3),
     ('dressing', 3),
     ('also,', 3),
     ('pork', 3),
     ('fun', 3),
     ('steaks', 3),
     ('egg', 3),
     ('treated', 3),
     ('new', 3),
     ('quick', 3),
     ('area', 3),
     ('eaten', 3),
     ('vegetables', 3),
     ('watched', 3),
     ('lot', 3),
     ('tables', 3),
     ('us.', 3),
     ('brought', 3),
     ('attentive', 3),
     ('meal.', 3),
     ('perfectly', 3),
     ('seated', 3),
     ('thai', 3),
     ('authentic', 3),
     ('return.', 3),
     ('amount', 3),
     ('prices.', 3),
     ("couldn't", 3),
     ('beef', 3),
     ('bland.', 3),
     ('these', 3),
     ('stay', 3),
     ('shrimp', 3),
     ('frozen', 3),
     ('wings', 3),
     ('burgers', 3),
     ('used', 3),
     ('see', 3),
     ('those', 3),
     ('literally', 3),
     ('table', 3),
     ('ask', 3),
     ('needs', 3),
     ('average', 3),
     ('dog', 3),
     ('one.', 3),
     ('location', 3),
     ('3', 3),
     ('again', 3),
     ('place!', 3),
     ('find', 3),
     ('soon', 3),
     ('twice', 3),
     ('delicious', 3),
     ('each', 3),
     ('extremely', 3),
     ('slow.', 3),
     ('wanted', 3),
     ('drive', 3),
     ('things', 3),
     ('patio', 3),
     ('sick', 3),
     ('wine', 3),
     ('spot', 3),
     ('hit', 3),
     ('again!', 3),
     ('damn', 3),
     ('right', 3),
     ('rolls', 3),
     ('atmosphere.', 3),
     ('bill', 3),
     ('believe', 3),
     ('duck', 3),
     ('day', 3),
     ('years', 3),
     ('disappointing', 3),
     ('town', 3),
     ('cold', 3),
     ('room', 3),
     ('night.', 3),
     ('small', 3),
     ('wall', 3),
     ('40', 3),
     ('good!', 3),
     ('place,', 3),
     ('wrong', 3),
     ('say,', 3),
     ('liked', 3),
     ('maybe', 3),
     ('soon.', 3),
     ('nothing', 3),
     ('unless', 3),
     ('back!', 3),
     ('while', 3),
     ('kind', 3),
     ('outside', 3),
     ('soup', 3),
     ('cool', 3),
     ('last', 3),
     ('such', 3),
     ('wife', 3),
     ('either.', 3),
     ('leave', 3),
     ('course', 3),
     ('guess', 3),
     ('pho', 3),
     ('oh', 3),
     ('edible.', 3),
     ('done', 3),
     ('half', 3),
     ('totally', 3),
     ('everyone', 3),
     ('equally', 3),
     ('lobster', 3),
     ('trip', 3),
     ('salad.', 3),
     ('20', 3),
     ('management', 3),
     ('fresh.', 3),
     ('while.', 3),
     ('phoenix', 3),
     ('rare', 3),
     ('restaurants', 2),
     ('something', 2),
     ('checked', 2),
     ('use', 2),
     ('grilled', 2),
     ('italian', 2),
     ('pizza.', 2),
     ('nice,', 2),
     ('sliced', 2),
     ('pulled', 2),
     ('husband', 2),
     ('batter', 2),
     ('menu,', 2),
     ('awesome.', 2),
     ('creamy', 2),
     ('busy', 2),
     ('building', 2),
     ('atmosphere,', 2),
     ('interesting', 2),
     ('service!', 2),
     ('potatoes', 2),
     ('under', 2),
     ('seen', 2),
     ('serves', 2),
     ('1', 2),
     ('especially', 2),
     ('meat.', 2),
     ('bowl', 2),
     ('realized', 2),
     ("you're", 2),
     ('folks.', 2),
     ('mom', 2),
     ('pop', 2),
     ('bay', 2),
     ('area.', 2),
     ('hope', 2),
     ('regular', 2),
     ('stop', 2),
     ('experience,', 2),
     ('finish', 2),
     ('preparing', 2),
     ('indian', 2),
     ('comfortable.', 2),
     ('clean.', 2),
     ('generous', 2),
     ('dessert', 2),
     ('filling', 2),
     ('price', 2),
     ('bartender', 2),
     ('cooked.', 2),
     ('tasty!', 2),
     ('decent', 2),
     ('need', 2),
     ('die', 2),
     ('for.', 2),
     ('folks', 2),
     ('care', 2),
     ('subway', 2),
     ('offers', 2),
     ('serving', 2),
     ('roast', 2),
     ('soooo', 2),
     ('grossed', 2),
     ('extra', 2),
     ('in.', 2),
     ('pace.', 2),
     ('please', 2),
     ('stir', 2),
     ('sugary', 2),
     ('driest', 2),
     ('reasonably', 2),
     ("aren't", 2),
     ('crust', 2),
     ('disappointment', 2),
     ('only.', 2),
     ('overpriced', 2),
     ('taco', 2),
     ('flavor!', 2),
     ('single', 2),
     ('needed', 2),
     ('water', 2),
     ('refill', 2),
     ('finally', 2),
     ('beat', 2),
     ('nachos', 2),
     ('sad', 2),
     ('break', 2),
     ('ladies', 2),
     ('basically', 2),
     ('although', 2),
     ('meh.', 2),
     ('expected', 2),
     ('amazing!', 2),
     ('disappointed!', 2),
     ('stopped', 2),
     ('vibe', 2),
     ('started', 2),
     ('review', 2),
     ('though!', 2),
     ('however,', 2),
     ('recent', 2),
     ('seemed', 2),
     ('heat.', 2),
     ('presentation', 2),
     ('friend', 2),
     ('did.', 2),
     ('high', 2),
     ('charcoal', 2),
     ('fell', 2),
     ('ambiance.', 2),
     ('fairly', 2),
     ('suck,', 2),
     ('wait,', 2),
     ('promise', 2),
     ('disappoint.', 2),
     ('wasting', 2),
     ('rice', 2),
     ('looking', 2),
     ('become', 2),
     ('35', 2),
     ('minutes,', 2),
     ('yet', 2),
     ('overall', 2),
     ('immediately', 2),
     ('guy', 2),
     ('behind', 2),
     ('nice.', 2),
     ('cheap', 2),
     ('black', 2),
     ('ambience', 2),
     ('music', 2),
     ('tender', 2),
     ('why', 2),
     ('huge', 2),
     ('without', 2),
     ('doubt', 2),
     ('had.', 2),
     ('seating', 2),
     ('stomach', 2),
     ('day.', 2),
     ('between', 2),
     ('toasted', 2),
     ('bathrooms', 2),
     ('sitting', 2),
     ('multiple', 2),
     ('list', 2),
     ('gave', 2),
     ('work', 2),
     ('valley', 2),
     ('return', 2),
     ('10', 2),
     ('town,', 2),
     ('side,', 2),
     ('pleasant', 2),
     ('rude', 2),
     ('times,', 2),
     ('lacked', 2),
     ('better.', 2),
     ('steak.', 2),
     ('tasty.', 2),
     ('deserves', 2),
     ('pleased', 2),
     ('desserts', 2),
     ('tip', 2),
     ('drink', 2),
     ('neighborhood', 2),
     ('cold,', 2),
     ('chef.', 2),
     ('spend', 2),
     ('elsewhere.', 2),
     ('job', 2),
     ('vegas,', 2),
     ('ten', 2),
     ('fast', 2),
     ('walked', 2),
     ('bars', 2),
     ('was.', 2),
     ('selection.', 2),
     (',', 2),
     ('bites', 2),
     ('roasted', 2),
     ('sat', 2),
     ('hot,', 2),
     ("you'd", 2),
     ('sushi,', 2),
     ('enjoyed', 2),
     ('look', 2),
     ('bug', 2),
     ('above', 2),
     ('greeted', 2),
     ('.', 2),
     ('me.', 2),
     ('worse', 2),
     ('boy', 2),
     ('rude.', 2),
     ('empty', 2),
     ('ended', 2),
     ('bad,', 2),
     ('lost', 2),
     ('heart', 2),
     ("isn't", 2),
     ('tuna', 2),
     ('atmosphere', 2),
     ('lovely', 2),
     ('all.', 2),
     ('honest.', 2),
     ('steak,', 2),
     ('!', 2),
     ('dealing', 2),
     ('rather', 2),
     ('flower', 2),
     ('impeccable.', 2),
     ('waited.', 2),
     ('cashier', 2),
     ('overpriced.', 2),
     ('large', 2),
     ('flavorful', 2),
     ('strip', 2),
     ('friendly,', 2),
     ('big', 2),
     ('seasoned', 2),
     ('inside.', 2),
     ('none', 2),
     ('them.', 2),
     ('orders', 2),
     ('dish.', 2),
     ('white', 2),
     ('yourself', 2),
     ('perfection', 2),
     ('tribute', 2),
     ('cant', 2),
     ('friends', 2),
     ('texture', 2),
     ('dish', 2),
     ('pancakes', 2),
     ('amazing,', 2),
     ('serve', 2),
     ('selections', 2),
     ('bar', 2),
     ('establishment', 2),
     ('favorite', 2),
     ('three', 2),
     ('dirty', 2),
     ('wait.', 2),
     ('needless', 2),
     ('set', 2),
     ('green', 2),
     ('special.', 2),
     ("we're", 2),
     ('hummus', 2),
     ('sucks.', 2),
     ('owners', 2),
     ('let', 2),
     ('today', 2),
     ('impressed.', 2),
     ('min', 2),
     ('combination', 2),
     ('cook', 2),
     ('beyond', 2),
     ('added', 2),
     ('mind', 2),
     ('quickly', 2),
     ('unfortunately,', 2),
     ('terrible.', 2),
     ('second', 2),
     ('one!', 2),
     ('attitudes', 2),
     ('received', 2),
     ('similar', 2),
     ('sashimi', 2),
     ('really,', 2),
     ('its', 2),
     ('awful.', 2),
     ('definately', 2),
     ('horrible', 2),
     ('towards', 2),
     ('tea', 2),
     ('part', 2),
     ('reasonable', 2),
     ('mall', 2),
     ('hours', 2),
     ('either', 2),
     ('bother', 2),
     ('tapas', 2),
     ('complain', 2),
     ('possible.', 2),
     ('chinese', 2),
     ('ok,', 2),
     ('several', 2),
     ('garlic', 2),
     ('fine', 2),
     ('year', 2),
     ('attentive.', 2),
     ('owner', 2),
     ('avoid', 2),
     ('crab', 2),
     ('legs', 2),
     ('satisfying', 2),
     ('flavor', 2),
     ('had!', 2),
     ('recently', 2),
     ('passed', 2),
     ("we've", 2),
     ('cafe', 2),
     ('boyfriend', 2),
     ('salsa', 2),
     ('brunch', 2),
     ('price.', 2),
     ('2007', 1),
     ('awesome!!', 1),
     ('special', 1),
     ('thanks', 1),
     ('dylan', 1),
     ('t.', 1),
     (':)', 1),
     ('tummy.', 1),
     ('whole', 1),
     ('underwhelming,', 1),
     ("we'll", 1),
     ('ninja', 1),
     ('times.', 1),
     ('bachi', 1),
     ('up!!', 1),
     ('note', 1),
     ('ventilation', 1),
     ('upgrading.', 1),
     ('pizza,', 1),
     ('reminded', 1),
     ('legit', 1),
     ('efficient.', 1),
     ('though', 1),
     ('looked', 1),
     ('overwhelmed', 1),
     ('needs,', 1),
     ('stayed', 1),
     ('professional', 1),
     ('end.', 1),
     ('dry,', 1),
     ('brisket', 1),
     ('pork.', 1),
     ('sashimi.', 1),
     ('cow', 1),
     ('tongue', 1),
     ('cheek', 1),
     ('downright', 1),
     ('you.', 1),
     ('rude...', 1),
     ('apologize', 1),
     ('anything.', 1),
     ('apparently', 1),
     ('heard', 1),
     ('chewy.', 1),
     ('madhouse.', 1),
     ('absolutley', 1),
     ('point', 1),
     ('finger', 1),
     ('item', 1),
     ('downside', 1),
     ('round', 1),
     ('flavorful.', 1),
     ('(it', 1),
     ('either)', 1),
     ('freezing', 1),
     ('portion', 1),
     ('huge!', 1),
     ('sandwich.', 1),
     ('classy/warm', 1),
     ('appetizers,', 1),
     ('succulent', 1),
     ('(baseball', 1),
     ('steak!!!!!', 1),
     ('disgusting!', 1),
     ('arrived', 1),
     ('quickly!', 1),
     ('decor.', 1),
     ('reason', 1),
     ('fill', 1),
     ('binge', 1),
     ('drinking', 1),
     ('carbs', 1),
     ('stomach.', 1),
     ('bought,', 1),
     ('house.', 1),
     ('rubber', 1),
     ('ahead', 1),
     ('warmer.', 1),
     ('breakfast,', 1),
     ('$4.00.', 1),
     ('flavorless', 1),
     ('describing', 1),
     ('tepid', 1),
     ('poisoning', 1),
     ('buffet.', 1),
     ('def', 1),
     ('fav', 1),
     ('sunglasses', 1),
     ('rolled', 1),
     ('eyes', 1),
     ('stayed...', 1),
     ('service:', 1),
     ('fan,', 1),
     ('eat,', 1),
     ('reminds', 1),
     ('shops', 1),
     ('san', 1),
     ('francisco', 1),
     ('sticks', 1),
     ('around.', 1),
     ("it'll", 1),
     ('trips', 1),
     ('phoenix!', 1),
     ('crawfish', 1),
     ('kids', 1),
     ('play', 1),
     ('nasty!', 1),
     ('managed', 1),
     ('blandest', 1),
     ('cuisine.', 1),
     ('cashew', 1),
     ('undercooked.', 1),
     ('host', 1),
     ('were,', 1),
     ('lack', 1),
     ('word,', 1),
     ('bitches!', 1),
     ('space', 1),
     ('tiny,', 1),
     ('elegantly', 1),
     ('decorated', 1),
     ('attention', 1),
     ('ignore', 1),
     ('(just', 1),
     ('sat,', 1),
     ('sun.', 1),
     ('prime', 1),
     ('rib', 1),
     ('section.', 1),
     ('anyways,', 1),
     ('more.', 1),
     ('batch', 1),
     ('thinking', 1),
     ('yay', 1),
     ('no!', 1),
     ('delicious,', 1),
     ('personable', 1),
     ('deal!', 1),
     ('greasy,', 1),
     ('unhealthy', 1),
     ('trimmed', 1),
     ('google', 1),
     ('imagine', 1),
     ('smashburger', 1),
     ('combos', 1),
     ('burger,', 1),
     ('fries,', 1),
     ('23', 1),
     ('deal.', 1),
     ('seems', 1),
     ('neat;', 1),
     ('bathroom', 1),
     ('trippy,', 1),
     ('dessert,', 1),
     ('pack', 1),
     ('to-go', 1),
     ('tiramisu', 1),
     ('cannoli', 1),
     ('thirty', 1),
     ('(although', 1),
     ('8', 1),
     ('vacant', 1),
     ('waiting).', 1),
     ('ever.', 1),
     ('healthy', 1),
     ('ethic', 1),
     ('less...', 1),
     ('interior', 1),
     ('beautiful.', 1),
     ('problem', 1),
     ('charge', 1),
     ('$11.99', 1),
     ('bigger', 1),
     ('sub', 1),
     ('(which', 1),
     ('vegetables).', 1),
     ('besides', 1),
     ("costco's.", 1),
     ('rave', 1),
     ('reviews', 1),
     ('here......what', 1),
     ('disappointment!', 1),
     ('extensive', 1),
     ('provides', 1),
     ('lots', 1),
     ('options', 1),
     ('breakfast.', 1),
     ('turkey', 1),
     ('jalapeno', 1),
     ('judge', 1),
     ('whether', 1),
     ('sides', 1),
     ('melted', 1),
     ('styrofoam', 1),
     ('fear', 1),
     ('sick.', 1),
     ('croutons', 1),
     ('homemade', 1),
     ('plus.', 1),
     ('visit.', 1),
     ('pricing', 1),
     ('concern', 1),
     ('mellow', 1),
     ('mushroom.', 1),
     ('noodles.', 1),
     ('margaritas', 1),
     ('contained', 1),
     ('eaten.', 1),
     ('bouchon.', 1),
     ('priced', 1),
     ('also!', 1),
     ('doughy', 1),
     ('flavorless.', 1),
     ('out,', 1),
     ('ensued.', 1),
     ('driving', 1),
     ...]




```python
positive_count.most_common()
```




    [('the', 239),
     ('and', 171),
     ('was', 114),
     ('i', 84),
     ('a', 84),
     ('is', 76),
     ('to', 65),
     ('this', 59),
     ('great', 46),
     ('in', 44),
     ('of', 39),
     ('good', 37),
     ('very', 36),
     ('for', 33),
     ('food', 33),
     ('with', 31),
     ('place', 31),
     ('my', 29),
     ('are', 29),
     ('we', 29),
     ('it', 28),
     ('you', 27),
     ('were', 27),
     ('on', 26),
     ('service', 26),
     ('so', 26),
     ('they', 25),
     ('had', 24),
     ('have', 23),
     ('our', 19),
     ('all', 18),
     ('be', 18),
     ('that', 17),
     ('really', 16),
     ('their', 16),
     ('not', 15),
     ('just', 15),
     ('time', 15),
     ('as', 15),
     ('will', 14),
     ('nice', 13),
     ('also', 13),
     ('friendly', 13),
     ('first', 13),
     ('here', 13),
     ('but', 12),
     ('at', 11),
     ('back', 11),
     ('best', 11),
     ('an', 11),
     ('good.', 11),
     ('loved', 10),
     ('he', 10),
     ('by', 10),
     ('restaurant', 10),
     ('love', 10),
     ('go', 10),
     ('some', 9),
     ('-', 9),
     ('place.', 9),
     ('one', 9),
     ('chicken', 9),
     ('what', 8),
     ('server', 8),
     ('staff', 8),
     ('fresh', 8),
     ('vegas', 8),
     ('only', 8),
     ('like', 8),
     ('definitely', 8),
     ('pretty', 8),
     ('service.', 8),
     ('always', 8),
     ('been', 7),
     ('us', 7),
     ("i'm", 7),
     ('came', 7),
     ('when', 7),
     ('perfect', 7),
     ('fantastic', 7),
     ('happy', 7),
     ('made', 7),
     ('get', 7),
     ('every', 6),
     ('has', 6),
     ('even', 6),
     ('amazing.', 6),
     ('awesome', 6),
     ('food.', 6),
     ('out', 6),
     ('or', 6),
     ('which', 6),
     ('if', 6),
     ('food,', 6),
     ('could', 6),
     ('come', 6),
     ('would', 6),
     ('wonderful', 6),
     ('say', 6),
     ('going', 5),
     ('order', 5),
     ('went', 5),
     ('your', 5),
     ("it's", 5),
     ('delicious!', 5),
     ('great.', 5),
     ('thing', 5),
     ('everything', 5),
     ('menu', 5),
     ('staff.', 5),
     ('bacon', 5),
     ('taste', 5),
     ('worth', 5),
     ('recommend', 5),
     ('good,', 5),
     ('pizza', 5),
     ('great,', 5),
     ('can', 5),
     ('here.', 5),
     ('delicious.', 5),
     ('experience.', 5),
     ('sauce', 5),
     ('5', 5),
     ('about', 5),
     ('still', 5),
     ('tried', 5),
     ('vegas.', 5),
     ('bread', 5),
     ('than', 5),
     ('more', 5),
     ('&', 5),
     ('excellent', 5),
     ('super', 4),
     ('little', 4),
     ('fantastic.', 4),
     ('any', 4),
     ('up', 4),
     ('buffet', 4),
     ('spot.', 4),
     ('inside', 4),
     ('quite', 4),
     ('breakfast', 4),
     ('check', 4),
     ('did', 4),
     ('there', 4),
     ('beer', 4),
     ('both', 4),
     ('want', 4),
     ('night', 4),
     ('from', 4),
     ("didn't", 4),
     ('waitress', 4),
     ('friendly.', 4),
     ('absolutely', 4),
     ('selection', 4),
     ('salad', 4),
     ('how', 4),
     ('potato', 4),
     ("i've", 4),
     ('ever', 4),
     ('amazing', 4),
     ('family', 4),
     ('never', 4),
     ('prices', 4),
     ('steak', 4),
     ('once', 4),
     ('spicy', 4),
     ('quality', 4),
     ('it!', 4),
     ('sweet', 4),
     ('that.', 4),
     ("can't", 4),
     ('since', 3),
     ('meal', 3),
     ('recommendation', 3),
     ('disappointed.', 3),
     ('me', 3),
     ('until', 3),
     ("won't", 3),
     ('fact', 3),
     ('because', 3),
     ('greek', 3),
     ('pork', 3),
     ('fun', 3),
     ('steaks', 3),
     ('next', 3),
     ('new', 3),
     ('left', 3),
     ('his', 3),
     ('table.', 3),
     ('quick', 3),
     ('way', 3),
     ('dining', 3),
     ('portions', 3),
     ('attentive', 3),
     ('well', 3),
     ('perfectly', 3),
     ('authentic', 3),
     ('prices.', 3),
     ('other', 3),
     ('fries', 3),
     ('may', 3),
     ('it.', 3),
     ('back.', 3),
     ('delicious', 3),
     ('seriously', 3),
     ('service,', 3),
     ('patio', 3),
     ('think', 3),
     ('ice', 3),
     ('cream', 3),
     ('spot', 3),
     ('hit', 3),
     ('seafood', 3),
     ('right', 3),
     ('clean', 3),
     ('make', 3),
     ('experience', 3),
     ('two', 3),
     ('dishes', 3),
     ('cooked', 3),
     ('duck', 3),
     ('town', 3),
     ('tasty', 3),
     ('feel', 3),
     ('thought', 3),
     ('night.', 3),
     ('good!', 3),
     ('wrong', 3),
     ('out.', 3),
     ('found', 3),
     ('sushi', 3),
     ('ordered', 3),
     ('them', 3),
     ('cool', 3),
     ('better', 3),
     ('now', 3),
     ('hot', 3),
     ('stars', 3),
     ('eat', 3),
     ('yummy', 2),
     ('many', 2),
     ('2', 2),
     ('thumbs', 2),
     ('grilled', 2),
     ('italian', 2),
     ('pizza.', 2),
     ('nice,', 2),
     ('tacos', 2),
     ('she', 2),
     ('awesome.', 2),
     ('dressing', 2),
     ('creamy', 2),
     ('atmosphere,', 2),
     ('interesting', 2),
     ('bowl', 2),
     ('after', 2),
     ("you're", 2),
     ('served', 2),
     ('folks.', 2),
     ('area.', 2),
     ('regular', 2),
     ('stop', 2),
     ('experience,', 2),
     ('comfortable.', 2),
     ('clean.', 2),
     ('generous', 2),
     ('taste.', 2),
     ('dessert', 2),
     ('bartender', 2),
     ('got', 2),
     ('cooked.', 2),
     ('tasty!', 2),
     ('die', 2),
     ('try', 2),
     ('soooo', 2),
     ('extra', 2),
     ('thai', 2),
     ('reasonably', 2),
     ('those', 2),
     ('ladies', 2),
     ('who', 2),
     ('manager', 2),
     ('too.', 2),
     ('amazing!', 2),
     ('stopped', 2),
     ('off', 2),
     ('vibe', 2),
     ('restaurant.', 2),
     ('though!', 2),
     ('give', 2),
     ("i'll", 2),
     ('ambiance.', 2),
     ('disappoint.', 2),
     ('twice', 2),
     ('each', 2),
     ('time.', 2),
     ('must', 2),
     ('ambience', 2),
     ('tender', 2),
     ('huge', 2),
     ('close', 2),
     ('side', 2),
     ('town,', 2),
     ('again!', 2),
     ('side,', 2),
     ('real', 2),
     ('eaten', 2),
     ('amount', 2),
     ('itself', 2),
     ('tasty.', 2),
     ('egg', 2),
     ('rolls', 2),
     ('pleased', 2),
     ('atmosphere.', 2),
     ('chef.', 2),
     ('ambiance', 2),
     ("i'd", 2),
     ('years', 2),
     ('fast', 2),
     ('away', 2),
     ('tell', 2),
     ('selection.', 2),
     ('again.', 2),
     ("you'd", 2),
     ('enjoyed', 2),
     ('small', 2),
     ('greeted', 2),
     ('.', 2),
     ('do', 2),
     ('home', 2),
     ('atmosphere', 2),
     ('lovely', 2),
     ('liked', 2),
     ('impeccable.', 2),
     ('large', 2),
     ('flavorful', 2),
     ('back!', 2),
     ('while', 2),
     ('friendly,', 2),
     ('kind', 2),
     ('seasoned', 2),
     ('inside.', 2),
     ('white', 2),
     ('salmon', 2),
     ('perfection', 2),
     ('tribute', 2),
     ('last', 2),
     ('cant', 2),
     ('such', 2),
     ('serve', 2),
     ('pho', 2),
     ('oh', 2),
     ('selections', 2),
     ('favorite', 2),
     ('dinner', 2),
     ('chips', 2),
     ('enough', 2),
     ('equally', 2),
     ('special.', 2),
     ('hummus', 2),
     ('sandwich', 2),
     ('added', 2),
     ('expect', 2),
     ('second', 2),
     ('know', 2),
     ('definately', 2),
     ('trip', 2),
     ('full', 2),
     ('no', 2),
     ('far', 2),
     ('reasonable', 2),
     ('considering', 2),
     ('wait', 2),
     ('while.', 2),
     ('tapas', 2),
     ('chinese', 2),
     ('seated', 2),
     ('pasta', 2),
     ('fine', 2),
     ('places', 2),
     ('phoenix', 2),
     ('wall', 2),
     ('day', 2),
     ('lunch', 2),
     ('overall,', 2),
     ('where', 2),
     ('satisfying', 2),
     ('down', 2),
     ('had!', 2),
     ('enjoy', 2),
     ('cafe', 2),
     ('boyfriend', 2),
     ('rare', 2),
     ('salsa', 2),
     ('few', 2),
     ('used', 2),
     ('2007', 1),
     ('awesome!!', 1),
     ('special', 1),
     ('thanks', 1),
     ('dylan', 1),
     ('t.', 1),
     (':)', 1),
     ('tummy.', 1),
     ('checked', 1),
     ('times.', 1),
     ('bachi', 1),
     ('burger', 1),
     ("friend's", 1),
     ('up!!', 1),
     ('pizza,', 1),
     ('reminded', 1),
     ('legit', 1),
     ('efficient.', 1),
     ('though', 1),
     ('looked', 1),
     ('overwhelmed', 1),
     ('needs,', 1),
     ('stayed', 1),
     ('professional', 1),
     ('end.', 1),
     ('cow', 1),
     ('tongue', 1),
     ('cheek', 1),
     ('absolutley', 1),
     ('point', 1),
     ('finger', 1),
     ('item', 1),
     ('menu,', 1),
     ('round', 1),
     ('4', 1),
     ('stars,', 1),
     ('flavorful.', 1),
     ('portion', 1),
     ('huge!', 1),
     ('sandwich.', 1),
     ('classy/warm', 1),
     ('appetizers,', 1),
     ('succulent', 1),
     ('(baseball', 1),
     ('steak!!!!!', 1),
     ('arrived', 1),
     ('quickly!', 1),
     ('decor.', 1),
     ('def', 1),
     ('coming', 1),
     ('fav', 1),
     ('running', 1),
     ('realized', 1),
     ('husband', 1),
     ('sunglasses', 1),
     ('service:', 1),
     ('fan,', 1),
     ('being', 1),
     ('eat,', 1),
     ('reminds', 1),
     ('mom', 1),
     ('pop', 1),
     ('shops', 1),
     ('san', 1),
     ('francisco', 1),
     ('bay', 1),
     ('hope', 1),
     ('sticks', 1),
     ('around.', 1),
     ("it'll", 1),
     ('trips', 1),
     ('phoenix!', 1),
     ('crawfish', 1),
     ('finish', 1),
     ('space', 1),
     ('tiny,', 1),
     ('elegantly', 1),
     ('decorated', 1),
     ('(just', 1),
     ('sat,', 1),
     ('sun.', 1),
     ('prime', 1),
     ('rib', 1),
     ('section.', 1),
     ('delicious,', 1),
     ('personable', 1),
     ('deal!', 1),
     ('trimmed', 1),
     ('combos', 1),
     ('burger,', 1),
     ('fries,', 1),
     ('23', 1),
     ('decent', 1),
     ('deal.', 1),
     ('sure', 1),
     ('dessert,', 1),
     ('need', 1),
     ('pack', 1),
     ('to-go', 1),
     ('tiramisu', 1),
     ('cannoli', 1),
     ('for.', 1),
     ('healthy', 1),
     ('ethic', 1),
     ('care', 1),
     ('less...', 1),
     ('interior', 1),
     ('beautiful.', 1),
     ('extensive', 1),
     ('provides', 1),
     ('lots', 1),
     ('options', 1),
     ('breakfast.', 1),
     ('serving', 1),
     ('jalapeno', 1),
     ('croutons', 1),
     ('homemade', 1),
     ('plus.', 1),
     ('in.', 1),
     ('pace.', 1),
     ('visit.', 1),
     ('priced', 1),
     ('also!', 1),
     ('driving', 1),
     ('tucson!', 1),
     ('great!', 1),
     ('goat', 1),
     ('taco', 1),
     ('skimp', 1),
     ('meat', 1),
     ('wow', 1),
     ('flavor!', 1),
     ('burgers', 1),
     ('7', 1),
     ('table', 1),
     ('fast.', 1),
     ('continue', 1),
     ('andddd', 1),
     ('date', 1),
     ('...', 1),
     ('highly', 1),
     ('anyone', 1),
     ('area', 1),
     ('(;', 1),
     ('here!!!"', 1),
     ('flair', 1),
     ('bartenders', 1),
     ('during', 1),
     ('late', 1),
     ('bank', 1),
     ('holiday', 1),
     ('rick', 1),
     ('steve', 1),
     ('wings', 1),
     ('drive.', 1),
     ('prompt.', 1),
     ('salads!', 1),
     ('perpared', 1),
     ('beautiful', 1),
     ('presentation', 1),
     ('3', 1),
     ('giant', 1),
     ('slices', 1),
     ('toast,', 1),
     ('lightly', 1),
     ('dusted', 1),
     ('powdered', 1),
     ('sugar.', 1),
     ('try,', 1),
     ('did.', 1),
     ('all,', 1),
     ('assure', 1),
     ('promise', 1),
     ('beer.', 1),
     ('ample', 1),
     ('thus', 1),
     ('far,', 1),
     ('visited', 1),
     ('nice.', 1),
     ('drive', 1),
     ('north', 1),
     ('scottsdale...', 1),
     ('bit', 1),
     ('disappointed!', 1),
     ('solid', 1),
     ('value,', 1),
     ('stay', 1),
     ('least', 1),
     ('once.', 1),
     ('music', 1),
     ('playing.', 1),
     ('yellow', 1),
     ('saffron', 1),
     ('seasoning.', 1),
     ('beans.', 1),
     ('military', 1),
     ('discount.', 1),
     ('it...friendly', 1),
     ('servers,', 1),
     ('imaginative', 1),
     ('menu.', 1),
     ('seating', 1),
     ('house,', 1),
     ('low-key,', 1),
     ('non-fancy,', 1),
     ('affordable', 1),
     ('prices,', 1),
     ('subway,', 1),
     ('subway', 1),
     ('meet', 1),
     ('expectations.', 1),
     ('pricey', 1),
     ('pay', 1),
     ('getting', 1),
     ('lot!', 1),
     ('wings,', 1),
     ('feeling', 1),
     ('satisfied.', 1),
     ('genuinely', 1),
     ('pleasant', 1),
     ('enthusiastic', 1),
     ('treat.', 1),
     ('portion.', 1),
     ('multiple', 1),
     ('times,', 1),
     ('chipotle,', 1),
     ('better.', 1),
     ('satifying', 1),
     ('damn', 1),
     ('steak.', 1),
     ('over', 1),
     ('power', 1),
     ('scallop,', 1),
     ('bathrooms', 1),
     ('decorated.', 1),
     ('han', 1),
     ('nan', 1),
     ('service!', 1),
     ('incredibly', 1),
     ('fish,', 1),
     ('prepared', 1),
     ('care.', 1),
     ('biscuits!!!', 1),
     ('one.', 1),
     ('up....way', 1),
     ('up.', 1),
     ('deserves', 1),
     ('stars.', 1),
     ('hour,', 1),
     ('list', 1),
     ('wines.', 1),
     ('believe', 1),
     ('belly', 1),
     ('hankering', 1),
     ('sushi.', 1),
     ('neighborhood', 1),
     ('gem', 1),
     ('!!!', 1),
     ('someone', 1),
     ('(me)', 1),
     ('likes', 1),
     ('cold,', 1),
     ('case,', 1),
     ('colder.', 1),
     ('incredible.', 1),
     ('hardest', 1),
     ('decision...', 1),
     ('honestly,', 1),
     ("m's", 1),
     ('supposed', 1),
     ('(amazing).', 1),
     ('dessert:', 1),
     ('panna', 1),
     ('cotta', 1),
     ('mussels', 1),
     ('wine', 1),
     ('reduction,', 1),
     ('tender,', 1),
     ('jewel', 1),
     ('las', 1),
     ('vegas,', 1),
     ('exactly', 1),
     ('hoping', 1),
     ('find', 1),
     ('nearly', 1),
     ('ten', 1),
     ('living', 1),
     ('walked', 1),
     ('stuffed', 1),
     ('people', 1),
     ('was.', 1),
     ('daily', 1),
     ('specials', 1),
     ('group.', 1),
     ('delish,', 1),
     ('incredible', 1),
     ('must-stop', 1),
     ('whenever', 1),
     ('put', 1),
     ('plastic', 1),
     ('containers', 1),
     ('opposed', 1),
     ('cramming', 1),
     ('paper', 1),
     ('takeout', 1),
     ('boxes.', 1),
     ('dos', 1),
     ('gringos!', 1),
     ('rich', 1),
     ('accordingly.', 1),
     ('pho!', 1),
     ('plethora', 1),
     ('salads', 1),
     ('sandwiches,', 1),
     ('gets', 1),
     ('seal', 1),
     ('approval.', 1),
     ('venture', 1),
     ('further', 1),
     ('sushi,', 1),
     ('mouthful,', 1),
     ('enjoyable', 1),
     ('relaxed', 1),
     ('venue', 1),
     ('couples', 1),
     ('groups', 1),
     ('etc.', 1),
     ('potatoes', 1),
     ('biscuit.', 1),
     ('promptly', 1),
     ('seated.', 1),
     (',', 1),
     ('phenomenal', 1),
     ('drink', 1),
     ('empty', 1),
     ('suggestions.', 1),
     ('personally', 1),
     ('hummus,', 1),
     ('pita,', 1),
     ('baklava,', 1),
     ('falafels', 1),
     ('baba', 1),
     ('ganoush', 1),
     ("(it's", 1),
     ('eggplant!).', 1),
     ('conclusion:', 1),
     ('filling', 1),
     ('meals.', 1),
     ('excellent.', 1),
     ('chow', 1),
     ('mein', 1),
     ('reasonable,', 1),
     ('flavors', 1),
     ('on,', 1),
     ('made,', 1),
     ('slaw', 1),
     ('drenched', 1),
     ('mayo.', 1),
     ('nachos', 1),
     ('have!', 1),
     ('place,', 1),
     ('donut', 1),
     ('place!', 1),
     ('including', 1),
     ('massive', 1),
     ('meatloaf', 1),
     ('sandwich,', 1),
     ('crispy', 1),
     ('wrap,', 1),
     ('delish', 1),
     ('tuna', 1),
     ('melt', 1),
     ('burgers.', 1),
     ('duo', 1),
     ('violinists', 1),
     ('playing', 1),
     ('songs', 1),
     ('requested.', 1),
     ('ribeye', 1),
     ('mesquite', 1),
     ('flavor.', 1),
     ('cute,', 1),
     ('quaint,', 1),
     ('simple,', 1),
     ('honest.', 1),
     ('100%', 1),
     ('recommended!', 1),
     ('steak,', 1),
     ('sides,', 1),
     ('wine,', 1),
     ('desserts.', 1),
     ("you'll", 1),
     ('impressed', 1),
     ('familiar,', 1),
     ('vegetables', 1),
     ('feels', 1),
     ('thai.', 1),
     ('outstanding.', 1),
     ('pleasure', 1),
     ('dealing', 1),
     ('him.', 1),
     ('glad', 1),
     ('enough..', 1),
     ('actually.', 1),
     ('service-check!', 1),
     ('proven', 1),
     ('dead', 1),
     ('bar,', 1),
     ('these', 1),
     ('twice.', 1),
     ('times', 1),
     ('soon.', 1),
     ('job', 1),
     ('handling', 1),
     ('rowdy', 1),
     ('heat.', 1),
     ('shrimp', 1),
     ('opinion', 1),
     ('entrees', 1),
     ('gc.', 1),
     ('venturing', 1),
     ('strip', 1),
     ('belly,', 1),
     ('return', 1),
     ('madison', 1),
     ('ironman,', 1),
     ('deliciously', 1),
     ('fry', 1),
     ('outside', 1),
     ('moist', 1),
     ('"to', 1),
     ('go"', 1),
     ('orders', 1),
     ('omg', 1),
     ('felt', 1),
     ('dish.', 1),
     ('toro', 1),
     ('tartare', 1),
     ('cavier', 1),
     ('extraordinary', 1),
     ('thinly', 1),
     ('sliced', 1),
     ('wagyu', 1),
     ('truffle.', 1),
     ('expert/connisseur', 1),
     ('topic.', 1),
     ('dishes,', 1),
     ('best,', 1),
     ('redeeming', 1),
     ('inexpensive.', 1),
     ('owner/chef,', 1),
     ('japanese', 1),
     ('dude!', 1),
     ('am', 1),
     ('review', 1),
     ('hereas', 1),
     ('event', 1),
     ('held', 1),
     ('funny.', 1),
     ('surprise!', 1),
     ('lordy,', 1),
     ('khao', 1),
     ('soi', 1),
     ('dish', 1),
     ('missed', 1),
     ('curry', 1),
     ('lovers!', 1),
     ("mom's", 1),
     ('multi-grain', 1),
     ('pumpkin', 1),
     ('pancakes', 1),
     ('pecan', 1),
     ('butter', 1),
     ('amazing,', 1),
     ('fluffy,', 1),
     ('desserts', 1),
     ('yummy.', 1),
     ('vinaigrette', 1),
     ('overall', 1),
     ('dish,', 1),
     ('spring', 1),
     ('try.', 1),
     ('nyc', 1),
     ('bagels,', 1),
     ('cheese,', 1),
     ('lox', 1),
     ('capers', 1),
     ('even.', 1),
     ('shawarrrrrrma!!!!!!', 1),
     ('sunday', 1),
     ('peanut', 1),
     ('pancake', 1),
     ('butter,', 1),
     ('bits', 1),
     ('top....very', 1),
     ('original', 1),
     ('wrapped', 1),
     ('dates.', 1),
     ('mandalay', 1),
     ('bay.', 1),
     ('lighting', 1),
     ('dark', 1),
     ('set', 1),
     ('mood.', 1),
     ('customize', 1),
     ('like,', 1),
     ('usual', 1),
     ('eggplant', 1),
     ('green', 1),
     ('bean', 1),
     ('stir', 1),
     ('fry,', 1),
     ('everyone', 1),
     ('treated', 1),
     ('waiter', 1),
     ('attentive,', 1),
     ('informative.', 1),
     ('seasonal', 1),
     ('fruit', 1),
     ('peach', 1),
     ('puree.', 1),
     ('accident', 1),
     ('happier.', 1),
     ('classic', 1),
     ('maine', 1),
     ('lobster', 1),
     ('roll', 1),
     ('tasty,', 1),
     ('pita', 1),
     ('refreshing.', 1),
     ('delights,', 1),
     ('owners', 1),
     ('courteous.', 1),
     ('monster', 1),
     ('fried', 1),
     ('eggs', 1),
     ('favorite.', 1),
     ('buffets', 1),
     ('to.', 1),
     ('firehouse!!!!!', 1),
     ('deal', 1),
     ('included', 1),
     ('tastings', 1),
     ('drinks,', 1),
     ('jeff', 1),
     ('above', 1),
     ('beyond', 1),
     ('expected.', 1),
     ('atmosphere.1', 1),
     ('gratuity', 1),
     ('bill', 1),
     ('party', 1),
     ('larger', 1),
     ('6', 1),
     ('8,', 1),
     ('tip', 1),
     ('always,', 1),
     ('compliments', 1),
     ('update.....went', 1),
     ('setting.', 1),
     ('things', 1),
     ('diverse,', 1),
     ('priced.', 1),
     ('ever,', 1),
     ('maria', 1),
     ('day.', 1),
     ('joint', 1),
     ('tender.', 1),
     ('quickly', 1),
     ('become', 1),
     ('regular.', 1),
     ('vanilla', 1),
     ('smooth', 1),
     ('profiterole', 1),
     ('(choux)', 1),
     ('pastry', 1),
     ('enough.', 1),
     ('mention', 1),
     ('combination', 1),
     ('pears,', 1),
     ('almonds', 1),
     ('big', 1),
     ('winner!', 1),
     ('summer,', 1),
     ('dine', 1),
     ('charming', 1),
     ('outdoor', 1),
     ('delightful.', 1),
     ('roast', 1),
     ('beef', 1),
     ('tasted', 1),
     ('say.', 1),
     ('pulled', 1),
     ('waiter,', 1),
     ('helpful', 1),
     ('kept', 1),
     ('bloddy', 1),
     ("mary's", 1),
     ('coming.', 1),
     ('four', 1),
     ('guy', 1),
     ('blue', 1),
     ('shirt', 1),
     ...]




```python
total_words=0
for pair in total_count.most_common():
  total_words+=pair[1]

print(total_words)
```

    8751
    

# Calculating probability of occurrence:
### e.g.: P[“the”] = num of documents containing ‘the’ / num of all documents



```python
prob_all_docs={}
# stop_words=['','the','a','and','of','is','to','this','was','in','that','it','for','as','with','are','on','This','i']
stop_words=['']
for pair in total_count.most_common():
  count=0
  # print(pair[0])
  if pair[0].lower() not in stop_words:
    for sentence in train_list:
      if pair[0].lower() in sentence[0].lower():
        # print(sentence[0])
        count=count+1
    prob_all_docs[pair[0]]=count/len(train_list)
print(prob_all_docs)  
```

    {'the': 0.5475, 'and': 0.34125, 'was': 0.29375, 'i': 0.9275, 'a': 0.9425, 'to': 0.25375, 'is': 0.32125, 'this': 0.13125, 'of': 0.13375, 'not': 0.13, 'for': 0.12875, 'it': 0.3425, 'in': 0.4575, 'food': 0.13375, 'we': 0.17625, 'place': 0.09875, 'my': 0.08125, 'very': 0.08875, 'be': 0.21, 'so': 0.1875, 'that': 0.0675, 'with': 0.06625, 'had': 0.06875, 'service': 0.09125, 'good': 0.09125, 'were': 0.05875, 'they': 0.06, 'have': 0.06125, 'at': 0.3475, 'are': 0.09, 'you': 0.075, 'great': 0.06625, 'but': 0.065, 'on': 0.30875, 'our': 0.08, 'like': 0.045, 'will': 0.03875, 'just': 0.04, 'as': 0.4425, 'here': 0.10625, 'go': 0.17625, 'back': 0.06, 'all': 0.1275, 'time': 0.0575, 'really': 0.03375, 'their': 0.03125, 'if': 0.0425, 'never': 0.02875, 'would': 0.03125, 'an': 0.475, 'there': 0.0325, 'only': 0.02875, 'been': 0.02375, 'what': 0.02625, "don't": 0.02625, 'one': 0.07, 'your': 0.02625, 'by': 0.02625, 'out': 0.06625, 'no': 0.19875, 'get': 0.03625, '-': 0.04125, 'from': 0.02, 'also': 0.02625, 'food.': 0.02375, 'did': 0.0375, "won't": 0.02125, "i'm": 0.02125, 'came': 0.0225, 'when': 0.0225, 'good.': 0.02125, 'going': 0.02, 'some': 0.0425, 'he': 0.63125, 'or': 0.2775, 'got': 0.02, 'up': 0.0525, 'ever': 0.07875, 'more': 0.02375, 'definitely': 0.02125, 'which': 0.02125, 'chicken': 0.01875, 'us': 0.1875, 'pretty': 0.01875, 'eat': 0.1275, 'first': 0.01875, 'nice': 0.02375, "i've": 0.0175, 'than': 0.0175, 'about': 0.0175, 'it.': 0.02375, 'could': 0.02, 'friendly': 0.025, 'even': 0.02, 'service.': 0.01625, 'made': 0.01875, 'restaurant': 0.02625, 'how': 0.025, 'back.': 0.01625, 'much': 0.01625, 'server': 0.02, 'again.': 0.01625, "it's": 0.015, 'better': 0.01875, 'best': 0.0225, 'minutes': 0.01625, 'place.': 0.015, "didn't": 0.015, 'quality': 0.015, 'can': 0.03375, 'has': 0.015, 'other': 0.02875, 'went': 0.015, 'me': 0.25875, 'staff': 0.02, 'bad': 0.02125, 'any': 0.035, 'being': 0.01375, '&': 0.01125, 'love': 0.03625, 'think': 0.01375, 'loved': 0.0125, 'because': 0.0125, "wasn't": 0.0125, 'feel': 0.015, 'vegas': 0.02375, 'after': 0.01375, 'worth': 0.0125, 'still': 0.0125, 'always': 0.01375, 'say': 0.01875, 'know': 0.015, 'probably': 0.01, 'little': 0.01125, 'want': 0.01375, 'wait': 0.04125, 'fresh': 0.01625, 'too': 0.02625, 'way': 0.04, 'pizza': 0.015, "can't": 0.01125, 'come': 0.01625, 'do': 0.08, 'order': 0.02625, 'sushi': 0.01375, 'coming': 0.01125, 'recommend': 0.0175, 'give': 0.01125, 'down': 0.01875, 'worst': 0.0125, 'she': 0.02125, 'night': 0.0125, 'fries': 0.01, 'food,': 0.01, 'taste': 0.02625, 'bit': 0.01625, 'two': 0.01, 'over': 0.035, 'here.': 0.02375, 'every': 0.0225, 'experience': 0.02375, 'next': 0.01, 'tried': 0.00875, 'sauce': 0.01, 'dining': 0.0075, 'thing': 0.0325, 'waited': 0.0075, 'perfect': 0.01625, 'fantastic': 0.01375, 'now': 0.02375, 'ordered': 0.01, 'slow': 0.01125, 'them': 0.0125, 'happy': 0.00875, 'absolutely': 0.00875, 'salad': 0.0175, 'times': 0.0125, 'experience.': 0.00875, 'make': 0.00875, 'dishes': 0.00875, 'another': 0.0075, 'took': 0.00875, 'enough': 0.0125, 'meal': 0.01375, 'off': 0.01625, 'super': 0.0075, 'many': 0.0075, '2': 0.01875, 'meat': 0.0125, 'amazing.': 0.00875, 'awesome': 0.01125, 'service,': 0.0075, 'then': 0.0125, 'said': 0.00625, 'chips': 0.0075, 'buffet': 0.01, 'quite': 0.00875, 'everything': 0.01, 'menu': 0.0125, 'getting': 0.01, 'good,': 0.00625, 'once': 0.01, 'felt': 0.0075, 'restaurant.': 0.0075, 'waitress': 0.0075, 'who': 0.00875, 'her': 0.15, 'hot': 0.01, "i'll": 0.0075, 'selection': 0.0125, 'wonderful': 0.0075, 'few': 0.0075, '5': 0.0125, 'potato': 0.01125, 'vegas.': 0.00875, 'enjoy': 0.01, 'poor': 0.0075, 'around': 0.00875, 'take': 0.01125, 'since': 0.0075, 'time.': 0.00625, 'all,': 0.01125, 'here!': 0.0075, 'disappointed.': 0.00625, 'lunch': 0.00875, 'tell': 0.00625, 'kept': 0.00625, 'far': 0.01, 'barely': 0.00625, 'bad.': 0.00625, 'left': 0.0075, 'table.': 0.00875, 'may': 0.0125, 'sure': 0.01, 'served': 0.00625, 'delicious!': 0.0075, 'great.': 0.00625, 'inside': 0.00875, 'breakfast': 0.01, 'check': 0.01125, 'should': 0.0075, 'expect': 0.01125, 'up.': 0.00875, 'beer': 0.0075, 'both': 0.00875, 'staff.': 0.00625, 'bacon': 0.00625, 'hard': 0.00875, 'amazing': 0.02375, 'best.': 0.00625, 'great,': 0.00625, 'clean': 0.01, 'bring': 0.00625, 'asked': 0.00625, 'seafood': 0.00625, 'delicious.': 0.00625, 'spicy': 0.0075, '30': 0.0075, 'tasty': 0.0125, 'family': 0.00625, 'thought': 0.00625, 'prices': 0.01125, 'steak': 0.015, 'found': 0.0075, 'tasted': 0.00625, 'dinner': 0.0075, 'bread': 0.00625, 'excellent': 0.00875, 'waste': 0.00625, 'burger': 0.015, 'zero': 0.00375, 'stars': 0.0125, 'mediocre': 0.00625, 'terrible': 0.01125, 'fantastic.': 0.005, 'fact': 0.005, 'cold.': 0.005, 'before': 0.00625, 'places': 0.005, 'bland': 0.0125, 'spot.': 0.005, 'running': 0.005, 'his': 0.13625, 'try': 0.01125, 'cream': 0.0075, 'waiter': 0.0075, 'pay': 0.00625, 'portions': 0.005, 'taste.': 0.005, 'warm': 0.01125, 'well': 0.01125, 'itself': 0.005, 'sandwich': 0.00875, 'away': 0.00625, 'fried': 0.0075, 'overall,': 0.005, 'disappointed': 0.01375, 'where': 0.00875, 'manager': 0.005, 'friendly.': 0.005, 'too.': 0.005, 'eating': 0.00875, 'cooked': 0.0125, 'servers': 0.00625, 'anytime': 0.005, 'impressed': 0.0075, 'talk': 0.005, 'seriously': 0.00625, 'must': 0.00625, 'least': 0.005, 'money': 0.00625, 'close': 0.00625, 'am': 0.09875, 'waiting': 0.00625, 'ice': 0.155, 'side': 0.03, 'real': 0.04375, 'flavor.': 0.005, 'hour': 0.00875, 'someone': 0.005, 'ambiance': 0.00875, "i'd": 0.005, 'customer': 0.00625, 'people': 0.00875, 'considering': 0.005, 'home': 0.0075, 'out.': 0.005, 'deal': 0.01, 'there.': 0.005, 'long': 0.00625, 'it!': 0.00625, 'sweet': 0.00625, 'that.': 0.00625, 'business': 0.00625, 'full': 0.0075, 'tasteless.': 0.005, 'well.': 0.005, 'pasta': 0.00375, 'live': 0.00875, 'recommendation': 0.00375, 'yummy': 0.005, 'total': 0.0075, 'back,': 0.00375, "wouldn't": 0.00375, "friend's": 0.005, 'thumbs': 0.00375, 'stars.': 0.00375, 'until': 0.00375, 'salmon': 0.00375, 'tacos': 0.005, 'salt': 0.005, 'fish': 0.00625, '4': 0.01125, 'stars,': 0.00375, 'greek': 0.0025, 'dressing': 0.00375, 'also,': 0.00375, 'pork': 0.005, 'fun': 0.005, 'steaks': 0.00375, 'egg': 0.01, 'treated': 0.00375, 'new': 0.00375, 'quick': 0.0075, 'area': 0.00625, 'eaten': 0.005, 'vegetables': 0.005, 'watched': 0.00375, 'lot': 0.00625, 'tables': 0.01, 'us.': 0.01375, 'brought': 0.00375, 'attentive': 0.0075, 'meal.': 0.00375, 'perfectly': 0.00375, 'seated': 0.00625, 'thai': 0.005, 'authentic': 0.00375, 'return.': 0.00375, 'amount': 0.00375, 'prices.': 0.00375, "couldn't": 0.00375, 'beef': 0.00375, 'bland.': 0.005, 'these': 0.00375, 'stay': 0.0075, 'shrimp': 0.00625, 'frozen': 0.00375, 'wings': 0.005, 'burgers': 0.005, 'used': 0.00625, 'see': 0.01125, 'those': 0.00375, 'literally': 0.00375, 'table': 0.0225, 'ask': 0.01125, 'needs': 0.005, 'average': 0.005, 'dog': 0.00375, 'one.': 0.0075, 'location': 0.005, '3': 0.01625, 'again': 0.02375, 'place!': 0.00375, 'find': 0.00375, 'soon': 0.00875, 'twice': 0.005, 'delicious': 0.02, 'each': 0.005, 'extremely': 0.00375, 'slow.': 0.00375, 'wanted': 0.00375, 'drive': 0.005, 'things': 0.005, 'patio': 0.00375, 'sick': 0.005, 'wine': 0.00625, 'spot': 0.01125, 'hit': 0.00625, 'again!': 0.00375, 'damn': 0.00375, 'right': 0.005, 'rolls': 0.00375, 'atmosphere.': 0.005, 'bill': 0.00375, 'believe': 0.00375, 'duck': 0.00375, 'day': 0.01375, 'years': 0.00375, 'disappointing': 0.00625, 'town': 0.0075, 'cold': 0.01125, 'room': 0.00875, 'night.': 0.00375, 'small': 0.005, 'wall': 0.00375, '40': 0.005, 'good!': 0.005, 'place,': 0.00375, 'wrong': 0.00625, 'say,': 0.00375, 'liked': 0.00375, 'maybe': 0.00375, 'soon.': 0.00375, 'nothing': 0.00375, 'unless': 0.00375, 'back!': 0.00375, 'while': 0.0075, 'kind': 0.00375, 'outside': 0.005, 'soup': 0.005, 'cool': 0.00375, 'last': 0.0075, 'such': 0.00375, 'wife': 0.00375, 'either.': 0.00375, 'leave': 0.00625, 'course': 0.00375, 'guess': 0.00375, 'pho': 0.01125, 'oh': 0.00375, 'edible.': 0.005, 'done': 0.005, 'half': 0.00375, 'totally': 0.00375, 'everyone': 0.00375, 'equally': 0.00375, 'lobster': 0.00375, 'trip': 0.01, 'salad.': 0.00375, '20': 0.00625, 'management': 0.005, 'fresh.': 0.00375, 'while.': 0.00375, 'phoenix': 0.005, 'rare': 0.005, 'restaurants': 0.0025, 'something': 0.0025, 'checked': 0.0025, 'use': 0.0325, 'grilled': 0.0025, 'italian': 0.0025, 'pizza.': 0.0025, 'nice,': 0.0025, 'sliced': 0.0025, 'pulled': 0.0025, 'husband': 0.0025, 'batter': 0.0025, 'menu,': 0.0025, 'awesome.': 0.0025, 'creamy': 0.0025, 'busy': 0.00375, 'building': 0.0025, 'atmosphere,': 0.0025, 'interesting': 0.0025, 'service!': 0.0025, 'potatoes': 0.00375, 'under': 0.00875, 'seen': 0.0025, 'serves': 0.005, '1': 0.01625, 'especially': 0.0025, 'meat.': 0.0025, 'bowl': 0.0025, 'realized': 0.0025, "you're": 0.0025, 'folks.': 0.0025, 'mom': 0.00375, 'pop': 0.0025, 'bay': 0.00375, 'area.': 0.0025, 'hope': 0.005, 'regular': 0.005, 'stop': 0.00625, 'experience,': 0.0025, 'finish': 0.0025, 'preparing': 0.0025, 'indian': 0.0025, 'comfortable.': 0.0025, 'clean.': 0.0025, 'generous': 0.0025, 'dessert': 0.01, 'filling': 0.0025, 'price': 0.025, 'bartender': 0.00375, 'cooked.': 0.00375, 'tasty!': 0.0025, 'decent': 0.0025, 'need': 0.0125, 'die': 0.0075, 'for.': 0.0025, 'folks': 0.005, 'care': 0.00375, 'subway': 0.0025, 'offers': 0.0025, 'serving': 0.0025, 'roast': 0.005, 'soooo': 0.005, 'grossed': 0.0025, 'extra': 0.00375, 'in.': 0.02, 'pace.': 0.0025, 'please': 0.00625, 'stir': 0.0025, 'sugary': 0.0025, 'driest': 0.0025, 'reasonably': 0.0025, "aren't": 0.0025, 'crust': 0.00375, 'disappointment': 0.00375, 'only.': 0.0025, 'overpriced': 0.005, 'taco': 0.0075, 'flavor!': 0.0025, 'single': 0.0025, 'needed': 0.0025, 'water': 0.005, 'refill': 0.0025, 'finally': 0.0025, 'beat': 0.0025, 'nachos': 0.0025, 'sad': 0.005, 'break': 0.01375, 'ladies': 0.0025, 'basically': 0.0025, 'although': 0.00375, 'meh.': 0.0025, 'expected': 0.00375, 'amazing!': 0.005, 'disappointed!': 0.0025, 'stopped': 0.0025, 'vibe': 0.0025, 'started': 0.0025, 'review': 0.0075, 'though!': 0.0025, 'however,': 0.0025, 'recent': 0.005, 'seemed': 0.0025, 'heat.': 0.0025, 'presentation': 0.0025, 'friend': 0.03875, 'did.': 0.0025, 'high': 0.00625, 'charcoal': 0.0025, 'fell': 0.00375, 'ambiance.': 0.0025, 'fairly': 0.0025, 'suck,': 0.0025, 'wait,': 0.0025, 'promise': 0.0025, 'disappoint.': 0.0025, 'wasting': 0.0025, 'rice': 0.02875, 'looking': 0.0025, 'become': 0.0025, '35': 0.0025, 'minutes,': 0.0025, 'yet': 0.0025, 'overall': 0.0075, 'immediately': 0.00375, 'guy': 0.0025, 'behind': 0.0025, 'nice.': 0.0025, 'cheap': 0.0025, 'black': 0.0025, 'ambience': 0.0025, 'music': 0.0025, 'tender': 0.01, 'why': 0.00375, 'huge': 0.00375, 'without': 0.0025, 'doubt': 0.0025, 'had.': 0.0025, 'seating': 0.0025, 'stomach': 0.00375, 'day.': 0.0025, 'between': 0.0025, 'toasted': 0.0025, 'bathrooms': 0.0025, 'sitting': 0.0025, 'multiple': 0.0025, 'list': 0.005, 'gave': 0.0025, 'work': 0.005, 'valley': 0.0025, 'return': 0.00625, '10': 0.00625, 'town,': 0.0025, 'side,': 0.0025, 'pleasant': 0.0025, 'rude': 0.0075, 'times,': 0.0025, 'lacked': 0.0025, 'better.': 0.0025, 'steak.': 0.0025, 'tasty.': 0.0025, 'deserves': 0.0025, 'pleased': 0.00375, 'desserts': 0.00375, 'tip': 0.005, 'drink': 0.00625, 'neighborhood': 0.0025, 'cold,': 0.0025, 'chef.': 0.0025, 'spend': 0.00375, 'elsewhere.': 0.0025, 'job': 0.00375, 'vegas,': 0.0025, 'ten': 0.02875, 'fast': 0.015, 'walked': 0.0025, 'bars': 0.0025, 'was.': 0.0025, 'selection.': 0.0025, ',': 0.2925, 'bites': 0.0025, 'roasted': 0.0025, 'sat': 0.01, 'hot,': 0.0025, "you'd": 0.0025, 'sushi,': 0.0025, 'enjoyed': 0.0025, 'look': 0.0075, 'bug': 0.00125, 'above': 0.0025, 'greeted': 0.0025, '.': 0.83625, 'me.': 0.015, 'worse': 0.00125, 'boy': 0.00625, 'rude.': 0.00375, 'empty': 0.00375, 'ended': 0.005, 'bad,': 0.0025, 'lost': 0.0025, 'heart': 0.00375, "isn't": 0.0025, 'tuna': 0.00625, 'atmosphere': 0.01, 'lovely': 0.0025, 'all.': 0.00375, 'honest.': 0.0025, 'steak,': 0.0025, '!': 0.17, 'dealing': 0.0025, 'rather': 0.0025, 'flower': 0.0025, 'impeccable.': 0.0025, 'waited.': 0.0025, 'cashier': 0.0025, 'overpriced.': 0.0025, 'large': 0.005, 'flavorful': 0.00375, 'strip': 0.00375, 'friendly,': 0.0025, 'big': 0.005, 'seasoned': 0.0025, 'inside.': 0.0025, 'none': 0.00375, 'them.': 0.0025, 'orders': 0.0025, 'dish.': 0.0025, 'white': 0.0025, 'yourself': 0.0025, 'perfection': 0.0025, 'tribute': 0.00125, 'cant': 0.00375, 'friends': 0.00375, 'texture': 0.0025, 'dish': 0.015, 'pancakes': 0.0025, 'amazing,': 0.0025, 'serve': 0.0325, 'selections': 0.0025, 'bar': 0.02, 'establishment': 0.0025, 'favorite': 0.00375, 'three': 0.00125, 'dirty': 0.00375, 'wait.': 0.0025, 'needless': 0.0025, 'set': 0.005, 'green': 0.0025, 'special.': 0.0025, "we're": 0.0025, 'hummus': 0.00375, 'sucks.': 0.0025, 'owners': 0.0025, 'let': 0.015, 'today': 0.0025, 'impressed.': 0.0025, 'min': 0.04, 'combination': 0.0025, 'cook': 0.01625, 'beyond': 0.0025, 'added': 0.0025, 'mind': 0.005, 'quickly': 0.00375, 'unfortunately,': 0.0025, 'terrible.': 0.0025, 'second': 0.0025, 'one!': 0.00375, 'attitudes': 0.0025, 'received': 0.0025, 'similar': 0.00375, 'sashimi': 0.00375, 'really,': 0.0025, 'its': 0.01125, 'awful.': 0.0025, 'definately': 0.0025, 'horrible': 0.00375, 'towards': 0.0025, 'tea': 0.02, 'part': 0.0075, 'reasonable': 0.00375, 'mall': 0.0075, 'hours': 0.0025, 'either': 0.00875, 'bother': 0.0025, 'tapas': 0.0025, 'complain': 0.005, 'possible.': 0.0025, 'chinese': 0.0025, 'ok,': 0.0025, 'several': 0.0025, 'garlic': 0.0025, 'fine': 0.0025, 'year': 0.00625, 'attentive.': 0.0025, 'owner': 0.0075, 'avoid': 0.00375, 'crab': 0.00125, 'legs': 0.00125, 'satisfying': 0.00375, 'flavor': 0.0175, 'had!': 0.0025, 'recently': 0.0025, 'passed': 0.0025, "we've": 0.0025, 'cafe': 0.00375, 'boyfriend': 0.0025, 'salsa': 0.00125, 'brunch': 0.00375, 'price.': 0.0025, '2007': 0.00125, 'awesome!!': 0.00125, 'special': 0.0075, 'thanks': 0.00125, 'dylan': 0.00125, 't.': 0.10125, ':)': 0.00125, 'tummy.': 0.00125, 'whole': 0.00125, 'underwhelming,': 0.00125, "we'll": 0.00125, 'ninja': 0.00125, 'times.': 0.00125, 'bachi': 0.00125, 'up!!': 0.00125, 'note': 0.0025, 'ventilation': 0.00125, 'upgrading.': 0.00125, 'pizza,': 0.00125, 'reminded': 0.00125, 'legit': 0.0025, 'efficient.': 0.00125, 'though': 0.015, 'looked': 0.00125, 'overwhelmed': 0.0025, 'needs,': 0.00125, 'stayed': 0.0025, 'professional': 0.00125, 'end.': 0.00125, 'dry,': 0.00125, 'brisket': 0.00125, 'pork.': 0.00125, 'sashimi.': 0.00125, 'cow': 0.00125, 'tongue': 0.00125, 'cheek': 0.00125, 'downright': 0.00125, 'you.': 0.00125, 'rude...': 0.00125, 'apologize': 0.00125, 'anything.': 0.00125, 'apparently': 0.00125, 'heard': 0.00125, 'chewy.': 0.00125, 'madhouse.': 0.00125, 'absolutley': 0.00125, 'point': 0.03, 'finger': 0.00125, 'item': 0.00125, 'downside': 0.00125, 'round': 0.01, 'flavorful.': 0.00125, '(it': 0.0025, 'either)': 0.00125, 'freezing': 0.00125, 'portion': 0.0075, 'huge!': 0.00125, 'sandwich.': 0.00125, 'classy/warm': 0.00125, 'appetizers,': 0.00125, 'succulent': 0.00125, '(baseball': 0.00125, 'steak!!!!!': 0.00125, 'disgusting!': 0.00125, 'arrived': 0.00375, 'quickly!': 0.00125, 'decor.': 0.00125, 'reason': 0.00875, 'fill': 0.0075, 'binge': 0.00125, 'drinking': 0.00125, 'carbs': 0.00125, 'stomach.': 0.00125, 'bought,': 0.00125, 'house.': 0.0025, 'rubber': 0.00125, 'ahead': 0.00125, 'warmer.': 0.00125, 'breakfast,': 0.00125, '$4.00.': 0.00125, 'flavorless': 0.0025, 'describing': 0.00125, 'tepid': 0.00125, 'poisoning': 0.00125, 'buffet.': 0.00125, 'def': 0.025, 'fav': 0.00625, 'sunglasses': 0.00125, 'rolled': 0.00125, 'eyes': 0.00125, 'stayed...': 0.00125, 'service:': 0.00125, 'fan,': 0.00125, 'eat,': 0.0075, 'reminds': 0.00125, 'shops': 0.00125, 'san': 0.01375, 'francisco': 0.00125, 'sticks': 0.0025, 'around.': 0.00125, "it'll": 0.00125, 'trips': 0.00125, 'phoenix!': 0.00125, 'crawfish': 0.00125, 'kids': 0.00125, 'play': 0.00375, 'nasty!': 0.00125, 'managed': 0.00125, 'blandest': 0.00125, 'cuisine.': 0.00125, 'cashew': 0.00125, 'undercooked.': 0.00125, 'host': 0.0025, 'were,': 0.00125, 'lack': 0.0075, 'word,': 0.00125, 'bitches!': 0.00125, 'space': 0.00125, 'tiny,': 0.00125, 'elegantly': 0.00125, 'decorated': 0.0025, 'attention': 0.00125, 'ignore': 0.0025, '(just': 0.00125, 'sat,': 0.00125, 'sun.': 0.00125, 'prime': 0.00125, 'rib': 0.02, 'section.': 0.00125, 'anyways,': 0.00125, 'more.': 0.0025, 'batch': 0.00125, 'thinking': 0.00125, 'yay': 0.00125, 'no!': 0.00125, 'delicious,': 0.00125, 'personable': 0.00125, 'deal!': 0.00125, 'greasy,': 0.00125, 'unhealthy': 0.00125, 'trimmed': 0.00125, 'google': 0.00125, 'imagine': 0.0025, 'smashburger': 0.00125, 'combos': 0.00125, 'burger,': 0.00125, 'fries,': 0.00125, '23': 0.00125, 'deal.': 0.00125, 'seems': 0.00125, 'neat;': 0.00125, 'bathroom': 0.00375, 'trippy,': 0.00125, 'dessert,': 0.00125, 'pack': 0.0025, 'to-go': 0.00125, 'tiramisu': 0.00125, 'cannoli': 0.00125, 'thirty': 0.00125, '(although': 0.00125, '8': 0.00375, 'vacant': 0.00125, 'waiting).': 0.00125, 'ever.': 0.00125, 'healthy': 0.0025, 'ethic': 0.00125, 'less...': 0.00125, 'interior': 0.00125, 'beautiful.': 0.00125, 'problem': 0.00125, 'charge': 0.0025, '$11.99': 0.00125, 'bigger': 0.00125, 'sub': 0.00375, '(which': 0.00125, 'vegetables).': 0.00125, 'besides': 0.00125, "costco's.": 0.00125, 'rave': 0.00125, 'reviews': 0.0025, 'here......what': 0.00125, 'disappointment!': 0.00125, 'extensive': 0.00125, 'provides': 0.00125, 'lots': 0.00125, 'options': 0.00125, 'breakfast.': 0.00125, 'turkey': 0.00125, 'jalapeno': 0.00125, 'judge': 0.00125, 'whether': 0.00125, 'sides': 0.00375, 'melted': 0.00125, 'styrofoam': 0.00125, 'fear': 0.00125, 'sick.': 0.00125, 'croutons': 0.00125, 'homemade': 0.00125, 'plus.': 0.00125, 'visit.': 0.00125, 'pricing': 0.00125, 'concern': 0.00125, 'mellow': 0.00125, 'mushroom.': 0.00125, 'noodles.': 0.00125, 'margaritas': 0.00125, 'contained': 0.00125, 'eaten.': 0.00125, 'bouchon.': 0.00125, 'priced': 0.0075, 'also!': 0.00125, 'doughy': 0.00125, 'flavorless.': 0.00125, 'out,': 0.00125, 'ensued.': 0.00125, 'driving': 0.00125, 'tucson!': 0.00125, 'great!': 0.0025, 'shocked': 0.00125, 'signs': 0.00125, 'indicate': 0.00125, 'cash': 0.005, 'getting.': 0.00125, 'goat': 0.00125, 'skimp': 0.00125, 'wow': 0.0025, 'employee': 0.0025, 'ok': 0.0425, 'insulted': 0.0025, 'disrespected.': 0.00125, 'gross!': 0.00125, 'does': 0.00125, 'movies': 0.00125, 'spinach': 0.00125, 'avocado': 0.00125, 'salad;': 0.00125, 'ingredients': 0.00125, '7': 0.005, 'fast.': 0.0025, 'accountant': 0.00125, 'screwed!': 0.00125, 'experiencing': 0.00125, 'underwhelming': 0.0025, 'relationship': 0.00125, 'parties': 0.00125, 'person': 0.00375, 'continue': 0.00125, 'andddd': 0.00125, 'date': 0.00375, '...': 0.03625, 'highly': 0.00125, 'anyone': 0.0025, '(;': 0.00125, 'gyro': 0.0025, 'lettuce': 0.00125, 'blame': 0.00125, 'placed': 0.00125, 'door.': 0.00125, 'fly': 0.00125, 'apple': 0.0025, 'juice..': 0.00125, 'fly!!!!!!!!': 0.00125, 'sucked,': 0.00125, 'sucked': 0.0025, 'imagined.': 0.00125, 'hurry': 0.00125, 'here!!!"': 0.00125, 'flair': 0.00125, 'bartenders': 0.00125, 'paying': 0.00125, '$7.85': 0.00125, 'looks': 0.00125, "kid's": 0.00125, 'wienerschnitzel': 0.00125, 'idea': 0.00125, 'jerk.': 0.00125, 'thoroughly': 0.00125, 'during': 0.00125, 'late': 0.01, 'bank': 0.00125, 'holiday': 0.00125, 'rick': 0.0025, 'steve': 0.00125, 'editing': 0.00125, 'drive.': 0.00125, 'prompt.': 0.00125, 'particular': 0.00125, 'salads!': 0.00125, 'chipolte': 0.00125, 'ranch': 0.0025, 'dipping': 0.00125, 'sause': 0.00125, 'tasteless,': 0.00125, 'thin': 0.0475, 'watered': 0.00125, 'insulted.': 0.00125, 'perpared': 0.00125, 'beautiful': 0.00375, 'giant': 0.00125, 'slices': 0.00125, 'toast,': 0.00125, 'lightly': 0.00125, 'dusted': 0.00125, 'powdered': 0.00125, 'sugar.': 0.00125, 'bloody': 0.00125, 'mary.': 0.00125, 'try,': 0.00125, 'assure': 0.00125, 'hopes': 0.00125, 'grill,': 0.00125, 'unfortunately': 0.00375, 'flat,': 0.00125, 'flat.': 0.00125, 'simply': 0.00125, 'correction,': 0.00125, 'heimer': 0.00125, 'sucked.': 0.00125, 'soon!': 0.00125, '"mains,"': 0.00125, 'uninspired.': 0.00125, 'tragedy': 0.00125, 'struck.': 0.00125, 'cheated': 0.00125, 'opportunity': 0.00125, 'company.': 0.00125, 'beer.': 0.00125, 'despicable,': 0.00125, 'ample': 0.00125, 'self': 0.01, 'proclaimed': 0.00125, 'coffee': 0.0025, 'cafe,': 0.00125, 'wildly': 0.00125, 'thus': 0.0025, 'far,': 0.00125, 'visited': 0.00125, 'changing,': 0.00125, 'doing': 0.00125, 'shots': 0.00125, 'fireball': 0.00125, 'bar.': 0.00125, 'north': 0.00125, 'scottsdale...': 0.00125, 'hate': 0.0025, 'olives.': 0.00125, 'solid': 0.0025, 'value,': 0.00125, 'once.': 0.00125, 'playing.': 0.00125, 'yellow': 0.00125, 'saffron': 0.00125, 'seasoning.': 0.00125, 'beans.': 0.00125, 'military': 0.00125, 'discount.': 0.00125, 'it...friendly': 0.00125, 'servers,': 0.00125, 'imaginative': 0.00125, 'menu.': 0.00125, 'ache': 0.0025, 'rest': 0.03125, 'house,': 0.00125, 'low-key,': 0.00125, 'non-fancy,': 0.00125, 'affordable': 0.00125, 'prices,': 0.00125, 'connoisseur': 0.00125, 'difference': 0.00125, 'certainly': 0.00125, 'english': 0.00125, 'muffin': 0.00125, 'untoasted.': 0.00125, 'subway,': 0.00125, 'meet': 0.00125, 'expectations.': 0.00125, 'pricey': 0.00125, 'lot!': 0.00125, 'dirty-': 0.00125, 'seat': 0.01, 'covers': 0.00125, 'replenished': 0.00125, 'plain': 0.00625, 'yucky!!!': 0.00125, 'lukewarm,': 0.00125, 'wings,': 0.00125, 'feeling': 0.00125, 'satisfied.': 0.00125, 'ignored': 0.00125, 'hostess': 0.00125, 'myself.': 0.00125, 'trying': 0.00125, '(teeth': 0.00125, 'sore).': 0.00125, 'hospitality': 0.00125, 'industry': 0.00125, 'paradise': 0.00125, 'refrained': 0.00125, 'recommending': 0.00125, 'cibo': 0.00125, 'longer.': 0.00125, 'circumstances': 0.00125, 'to,': 0.0025, 'tops': 0.00125, 'list.': 0.00125, 'struggle': 0.00125, 'wave': 0.00125, 'minutes.': 0.00125, 'genuinely': 0.00125, 'enthusiastic': 0.00125, 'treat.': 0.00125, 'inconsiderate': 0.00125, 'management.': 0.00125, 'portion.': 0.00125, 'sweet,': 0.00125, 'enough,': 0.00125, 'overcooked?': 0.00125, 'chipotle,': 0.00125, 'satifying': 0.00125, 'rock': 0.00125, 'casino': 0.00125, 'before,': 0.00125, 'step': 0.00125, 'forward': 0.00125, 'power': 0.00125, 'scallop,': 0.00125, 'decorated.': 0.00125, 'han': 0.03, 'nan': 0.00375, 'help.': 0.00125, 'incredibly': 0.00125, 'fish,': 0.00125, 'prepared': 0.00125, 'care.': 0.00125, 'seated,': 0.00125, 'greatest': 0.00125, 'moods.': 0.00125, 'biscuits!!!': 0.00125, 'up....way': 0.00125, 'strange.': 0.00125, 'paid': 0.0025, 'job.': 0.00125, 'stinks': 0.00125, 'hour,': 0.00125, 'wines.': 0.00125, 'longer': 0.0025, 'arepas.': 0.00125, 'belly': 0.0025, 'hankering': 0.00125, 'sushi.': 0.00125, 'gem': 0.00625, '!!!': 0.01625, '(me)': 0.00125, 'likes': 0.00125, 'case,': 0.00125, 'colder.': 0.00125, 'incredible.': 0.00125, 'hardest': 0.00125, 'decision...': 0.00125, 'honestly,': 0.00125, "m's": 0.0025, 'supposed': 0.00125, '(amazing).': 0.00125, 'dessert:': 0.00125, 'panna': 0.00125, 'cotta': 0.00125, 'mussels': 0.00125, 'reduction,': 0.00125, 'tender,': 0.00125, 'not,': 0.00125, 'low': 0.02375, 'tolerance': 0.00125, 'people,': 0.00125, 'polite,': 0.00125, 'wash': 0.00125, 'otherwise!!': 0.00125, 'jewel': 0.00125, 'las': 0.015, 'exactly': 0.00125, 'hoping': 0.00125, 'nearly': 0.00125, 'living': 0.00125, 'stuffed': 0.0025, 'recommended': 0.0025, 'anyone!': 0.00125, 'terrible!': 0.0025, 'recall': 0.00125, 'charged': 0.00125, 'tap': 0.00375, 'water.': 0.00125, 'ians': 0.00125, 'daily': 0.00125, 'specials': 0.00125, 'group.': 0.00125, 'delish,': 0.00125, 'incredible': 0.0025, 'refused': 0.00125, 'anymore.': 0.00125, 'grandmother': 0.00125, 'bellagio': 0.00125, 'anticipated.': 0.00125, 'must-stop': 0.00125, 'whenever': 0.00125, 'disappointing!!!': 0.00125, 'put': 0.0025, 'plastic': 0.00125, 'containers': 0.00125, 'opposed': 0.00125, 'cramming': 0.00125, 'paper': 0.00375, 'takeout': 0.00125, 'boxes.': 0.00125, 'left.': 0.00125, 'temp.i': 0.00125, 'prepare': 0.0025, 'bare': 0.0075, 'hands,': 0.00125, 'gloves.everything': 0.00125, 'deep': 0.00125, 'oil.': 0.00125, 'dos': 0.00125, 'gringos!': 0.00125, 'rich': 0.00125, 'accordingly.': 0.00125, 'pho!': 0.00125, 'plethora': 0.00125, 'salads': 0.0025, 'sandwiches,': 0.00125, 'gets': 0.00125, 'seal': 0.00125, 'approval.': 0.00125, 'disgrace.': 0.00125, 'fair': 0.00375, 'venture': 0.00125, 'further': 0.0025, 'mouthful,': 0.00125, 'enjoyable': 0.00125, 'relaxed': 0.00125, 'venue': 0.00125, 'couples': 0.00125, 'groups': 0.00125, 'etc.': 0.00125, 'fella': 0.00125, 'huevos': 0.00125, 'rancheros': 0.00125, 'appealing.': 0.00125, 'biscuit.': 0.00125, 'possible': 0.00375, "they'd": 0.00125, 'showed': 0.00125, 'given': 0.00125, 'sure,': 0.00125, 'climbing': 0.00125, 'kitchen.': 0.00125, 'promptly': 0.00125, 'seated.': 0.00125, 'else.': 0.00125, 'spends': 0.00125, 'talking': 0.00125, 'themselves': 0.00125, 'officially': 0.00125, 'done.': 0.00125, 'bus': 0.0125, 'hand': 0.00875, 'generic.': 0.00125, 'gross.': 0.00125, 'phenomenal': 0.00125, 'consider': 0.0075, 'theft.': 0.00125, 'suggestions.': 0.00125, 'fast,': 0.0025, 'but,': 0.00125, 'order,': 0.00125, 'arrived.': 0.00125, 'personally': 0.00125, 'hummus,': 0.00125, 'pita,': 0.00125, 'baklava,': 0.00125, 'falafels': 0.00125, 'baba': 0.00125, 'ganoush': 0.00125, "(it's": 0.00125, 'eggplant!).': 0.00125, 'bagels': 0.0025, 'grocery': 0.00125, 'store.': 0.00125, 'conclusion:': 0.00125, 'meals.': 0.00125, 'excellent.': 0.00125, 'chow': 0.00125, 'mein': 0.00125, 'reasonable,': 0.00125, 'flavors': 0.00125, 'on,': 0.005, 'made,': 0.00125, 'slaw': 0.00125, 'drenched': 0.00125, 'mayo.': 0.00125, 'have!': 0.00125, 'donut': 0.00125, 'including': 0.00125, 'massive': 0.00125, 'meatloaf': 0.00125, 'sandwich,': 0.00125, 'crispy': 0.0025, 'wrap,': 0.00125, 'delish': 0.0025, 'melt': 0.0025, 'burgers.': 0.00125, 'duo': 0.00125, 'violinists': 0.00125, 'playing': 0.0025, 'songs': 0.00125, 'requested.': 0.00125, 'ribeye': 0.00125, 'mesquite': 0.00125, 'ri': 0.23375, 'style': 0.0025, 'calamari': 0.00125, 'joke.': 0.00125, 'par': 0.02125, "denny's,": 0.00125, "carly's": 0.00125, 'cute,': 0.00125, 'quaint,': 0.00125, 'simple,': 0.00125, '100%': 0.00125, 'recommended!': 0.00125, 'sides,': 0.00125, 'wine,': 0.00125, 'desserts.': 0.00125, "you'll": 0.00125, 'boot,': 0.00125, 'worries.': 0.00125, 'familiar,': 0.00125, 'honor': 0.00125, 'hut': 0.00125, 'coupons.': 0.00125, 'feels': 0.00125, 'thai.': 0.00125, 'money.': 0.00125, 'outstanding.': 0.00125, 'pleasure': 0.00125, 'him.': 0.00125, 'glad': 0.00125, 'letdown,': 0.00125, 'camelback': 0.00125, 'shop': 0.00375, 'cartel': 0.00125, 'coffee.': 0.00125, 'enough..': 0.00125, 'actually.': 0.00125, 'vegetarian': 0.00125, 'fare,': 0.00125, 'wrong:': 0.00125, 'burned': 0.00125, 'saganaki.': 0.00125, 'service-check!': 0.00125, 'proven': 0.00125, 'dead': 0.00125, 'bar,': 0.00125, 'twice.': 0.00125, 'mortified.': 0.00125, 'wayyy': 0.00125, 'honestly': 0.0025, 'blown': 0.00125, 'fails': 0.00125, 'deliver.': 0.00125, 'postinos,': 0.00125, 'waaaaaayyyyyyyyyy': 0.00125, 'rated': 0.00625, 'saying.': 0.00125, 'handling': 0.00125, 'rowdy': 0.00125, 'outta': 0.00125, 'opinion': 0.00125, 'entrees': 0.0025, 'gc.': 0.00125, 'pucks': 0.00125, 'disgust,': 0.00125, 'register.': 0.00125, 'turn': 0.0075, 'else': 0.00625, 'buying.': 0.00125, 'venturing': 0.00125, 'belly,': 0.00125, 'madison': 0.00125, 'ironman,': 0.00125, '"ya\'all".': 0.00125, 'flavor,': 0.00125, 'undercooked,': 0.00125, 'dry.': 0.00125, 'deliciously': 0.00125, 'fry': 0.0025, 'moist': 0.00125, 'descriptions': 0.00125, '"yum': 0.00125, 'yum': 0.00625, 'sauce"': 0.00125, '"eel': 0.00125, 'sauce",': 0.00125, '"spicy': 0.00125, 'mayo"...well': 0.00125, 'sauces': 0.00125, 'neither': 0.00125, 'burger.': 0.0025, 'dont': 0.00125, 'albondigas': 0.00125, 'tomato': 0.00125, 'meatballs.': 0.00125, '"to': 0.00125, 'go"': 0.00125, 'others.': 0.00125, 'omg': 0.0025, 'toro': 0.00125, 'tartare': 0.00125, 'cavier': 0.00125, 'extraordinary': 0.00125, 'thinly': 0.00125, 'wagyu': 0.00125, 'truffle.': 0.00125, 'mac': 0.005, 'expert/connisseur': 0.00125, 'topic.': 0.00125, 'dishes,': 0.00125, 'best,': 0.00125, 'favor': 0.005, 'redeeming': 0.00125, 'inexpensive.': 0.00125, 'owner/chef,': 0.00125, 'japanese': 0.00125, 'dude!': 0.00125, 'hereas': 0.00125, 'event': 0.00125, 'held': 0.00125, 'funny.': 0.00125, 'reading': 0.00125, 'surprise!': 0.00125, 'hated': 0.00125, '(coconut': 0.00125, 'shrimp),': 0.00125, 'meals,': 0.00125, 'nasty.': 0.00125, 'almost': 0.00125, 'excuse.': 0.00125, 'privileged': 0.00125, 'working/eating': 0.00125, 'lordy,': 0.00125, 'khao': 0.00125, 'soi': 0.00125, 'missed': 0.00125, 'curry': 0.00125, 'lovers!': 0.00125, "mom's": 0.00125, 'multi-grain': 0.00125, 'pumpkin': 0.00125, 'pecan': 0.00125, 'butter': 0.00375, 'fluffy,': 0.00125, 'yummy.': 0.00125, 'vinaigrette': 0.00125, 'dish,': 0.00125, 'college': 0.00125, 'cooking': 0.00125, 'class': 0.00375, 'disgraceful.': 0.00125, 'then,': 0.00125, "hadn't": 0.00125, 'wasted': 0.00125, 'life': 0.00125, 'there,': 0.00125, 'poured': 0.00125, 'wound': 0.00125, 'drawing': 0.00125, 'check.': 0.00125, 'spring': 0.00125, 'try.': 0.00125, 'though.': 0.00125, 'nyc': 0.00125, 'bagels,': 0.00125, 'cheese,': 0.00125, 'lox': 0.00125, 'capers': 0.00125, 'even.': 0.00125, "ryan's": 0.00125, 'edinburgh': 0.00125, 'revisiting.': 0.00125, "weren't": 0.00125, 'somewhat': 0.00125, 'shawarrrrrrma!!!!!!': 0.00125, 'different': 0.00125, 'occasions': 0.00125, 'medium': 0.00125, 'well,': 0.00125, 'bloodiest': 0.00125, 'piece': 0.00125, 'plate.': 0.00125, 'sunday': 0.00125, 'stupid': 0.00125, 'never,': 0.00125, 'oysters': 0.00125, 'were!': 0.00125, 'peanut': 0.0025, 'pancake': 0.00375, 'butter,': 0.00125, 'bits': 0.00125, 'top....very': 0.00125, 'original': 0.00125, 'wrapped': 0.0025, 'dates.': 0.00125, 'third,': 0.00125, 'cheese': 0.005, 'mandalay': 0.00125, 'bay.': 0.00125, 'thru': 0.00125, 'means': 0.0025, 'somehow': 0.00125, 'end': 0.0725, 'old,': 0.00375, 'chewy': 0.0025, 'way.': 0.0025, 'why.': 0.00125, 'dropped': 0.00125, 'ball.': 0.00125, 'luke': 0.00375, 'warm,': 0.0025, 'sever': 0.00375, 'overwhelmed.': 0.00125, 'dripping': 0.00125, 'grease,': 0.00125, 'mostly': 0.00125, 'lighting': 0.00125, 'dark': 0.00125, 'mood.': 0.00125, 'customize': 0.00125, 'like,': 0.00125, 'usual': 0.00125, 'eggplant': 0.0025, 'bean': 0.00375, 'fry,': 0.00125, 'airline': 0.00125, 'seriously.': 0.00125, 'appetite': 0.00125, 'instantly': 0.00125, 'gone.': 0.00125, 'attentive,': 0.00125, 'informative.': 0.00125, 'fried.': 0.00125, 'seasonal': 0.00125, 'fruit': 0.00125, 'peach': 0.00125, 'puree.': 0.00125, 'accident': 0.00125, 'happier.': 0.00125, 'classic': 0.00125, 'maine': 0.00125, 'roll': 0.00625, 'tasty,': 0.00125, 'pita': 0.00375, 'refreshing.': 0.00125, 'delights,': 0.00125, 'courteous.': 0.00125, 'time,': 0.00125, 'alone': 0.00125, 'hilarious,': 0.00125, 'christmas': 0.00125, 'eve': 0.105, 'remember': 0.00125, 'biggest': 0.00125, 'fail': 0.00375, 'entire': 0.00125, 'relocated': 0.00125, 'attack': 0.00125, 'grill': 0.005, 'downtown': 0.00125, 'flat-lined': 0.00125, 'excuse': 0.0025, 'milkshake,': 0.00125, 'chocolate': 0.00125, 'milk.': 0.00125, 'monster': 0.00125, 'eggs': 0.0025, 'favorite.': 0.00125, 'nutshell:': 0.00125, '1)': 0.00125, 'restaraunt': 0.00125, 'smells': 0.00125, 'market': 0.00125, 'sewer.': 0.00125, 'buffets': 0.00125, 'to.': 0.00125, 'hopefully': 0.00125, 'bodes': 0.00125, 'firehouse!!!!!': 0.00125, 'asking': 0.00125, 'order.': 0.00125, 'included': 0.00125, 'tastings': 0.00125, 'drinks,': 0.00125, 'jeff': 0.00125, 'expected.': 0.00125, 'attached': 0.00125, 'gas': 0.02625, 'station,': 0.00125, 'rarely': 0.00125, 'sign.': 0.00125, 'atmosphere.1': 0.00125, 'gratuity': 0.00125, 'party': 0.00125, 'larger': 0.00125, '6': 0.00125, '8,': 0.00125, 'always,': 0.00125, 'compliments': 0.00125, 'shower': 0.00125, 'rinse,': 0.00125, 'shower,': 0.00125, 'nude': 0.00125, 'see!': 0.00125, 'update.....went': 0.00125, 'empty.': 0.00125, 'ones': 0.0075, 'scene': 0.00125, 'it...definitely': 0.00125, 'setting.': 0.00125, 'diverse,': 0.00125, 'priced.': 0.00375, 'below': 0.00125, 'average.': 0.00125, 'ever,': 0.005, 'maria': 0.00125, 'forever': 0.00125, 'inflate,': 0.00125, 'smaller': 0.00125, 'grow': 0.00125, 'rapidly!': 0.00125, 'based': 0.00125, 'sub-par': 0.00125, 'effort': 0.00125, 'show': 0.00375, 'gratitude': 0.00125, 'parents': 0.00125, 'most': 0.00375, 'complaints': 0.0025, 'silently': 0.00125, 'chip': 0.01, 'sad...': 0.00125, 'count': 0.00375, 'box': 0.0025, '12.': 0.00125, 'car': 0.01, 'breaks': 0.00125, 'front': 0.00125, 'starving.': 0.00125, 'joint': 0.00125, 'final': 0.00375, 'blow!': 0.00125, 'sound': 0.00125, 'actual': 0.00375, 'disappointing.': 0.00125, 'surprised': 0.00125, 'article': 0.00125, 'read': 0.01, 'focused': 0.00125, 'spices': 0.00125, 'disgusting.': 0.00125, "shouldn't": 0.00125, 'eggs.': 0.00125, 'brownish': 0.00125, 'color': 0.00125, 'obviously': 0.00125, 'tender.': 0.00125, 'mean': 0.00375, 'famous': 0.00125, 'terrible!?!': 0.00125, '"real': 0.00125, 'traditional': 0.00125, 'hunan': 0.00125, 'style".': 0.00125, 'food/service': 0.00125, 'similarly,': 0.00125, 'delivery': 0.00125, 'man': 0.025, 'word': 0.0025, 'apology': 0.00125, '45': 0.00125, 'late.': 0.0025, 'regular.': 0.00125, 'concept': 0.00125, 'vanilla': 0.00125, 'smooth': 0.0025, 'profiterole': 0.00125, '(choux)': 0.00125, 'pastry': 0.00125, 'enough.': 0.0025, 'con:': 0.00125, 'spotty': 0.00125, 'it!!!!': 0.00125, 'mention': 0.00125, 'pears,': 0.00125, 'almonds': 0.00125, 'winner!': 0.00125, 'summer,': 0.00125, 'dine': 0.00125, 'charming': 0.00125, 'outdoor': 0.00125, 'delightful.': 0.00125, 'say.': 0.00125, 'lacking.': 0.00125, 'waiter,': 0.00125, 'helpful': 0.0025, 'bloddy': 0.00125, "mary's": 0.00125, 'coming.': 0.00125, 'soggy': 0.0025, 'four': 0.00125, 'blue': 0.00125, 'shirt': 0.00125, 'letting': 0.00125, 'far!!': 0.00125, 'ourselves.': 0.00125, 'leaves': 0.00125, 'desired.': 0.00125, 'tough': 0.00125, 'short': 0.00125, "haven't": 0.00125, 'gone': 0.0025, 'now!': 0.00125, '*': 0.0025, 'sour': 0.00125, 'soups': 0.00125, 'stars!': 0.00125, 'lovers,': 0.00125, "let's": 0.00125, 'honest': 0.00625, 'yama': 0.00125, 'customers,': 0.00125, 'customers': 0.00125, 'tots': 0.00125, 'onion': 0.00125, 'rings': 0.0025, 'bland...': 0.00125, 'liking': 0.00125, 'number': 0.00125, 'reasons': 0.00125, 'reviewing..': 0.00125, 'that...': 0.00125, 'dressed': 0.00125, 'rudely!': 0.00125, 'bite,': 0.00125, 'hooked.': 0.00125, 'sorry,': 0.00125, ':(': 0.00125, 'setting,': 0.00125, 'douchey': 0.00125, 'indoor': 0.00125, 'garden': 0.00125, 'biscuits.': 0.00125, 'complaints!': 0.00125, 'ayce': 0.00125, 'regularly,': 0.00125, 'pissd.': 0.00125, 'stale!': 0.00125, 'hawaiian': 0.00125, 'breeze,': 0.00125, 'mango': 0.00125, 'magic,': 0.00125, 'pineapple': 0.00125, 'delight': 0.005, 'smoothies': 0.00125, "they're": 0.00125, 'better,': 0.00125, 'dedicated': 0.00125, 'boba': 0.00125, 'spots,': 0.00125, 'jenni': 0.00125, 'pho.': 0.00125, 'located': 0.0025, 'crystals': 0.00125, 'shopping': 0.00125, 'aria.': 0.00125, 'time!': 0.00125, 'thats': 0.00125, 'scallop': 0.0025, 'appalling': 0.00125, 'value': 0.0025, 'pizzas': 0.00125, 'later': 0.00125, 'did!': 0.00125, 'unbelievable': 0.00125, 'bargain!': 0.00125, 'outshining': 0.00125, 'halibut.': 0.00125, 'mess': 0.00125, 'amazing!!!': 0.00125, "joey's": 0.00125, 'voted': 0.00125, 'readers': 0.00125, 'magazine.': 0.00125, 'point,': 0.00125, 'figured': 0.00125, 'joke': 0.0025, 'making': 0.00125, 'publicly': 0.00125, 'loudly': 0.00125, 'known.': 0.00125, '*heart*': 0.00125, 'allergy': 0.00125, 'warnings': 0.00125, 'clue': 0.00125, 'meals': 0.00375, 'contain': 0.00375, 'peanuts.': 0.00125, 'companions': 0.00125, 'told': 0.00125, 'me...everything': 0.00125, 'weekly': 0.00125, 'haunt,': 0.00125, 'greedy': 0.00125, 'corporation': 0.00125, 'dime': 0.00125, 'me!': 0.00375, 'despite': 0.00125, 'rate': 0.00875, 'businesses,': 0.00125, 'actually': 0.0025, 'star.': 0.00125, 'anything': 0.0025, 'veggitarian': 0.00125, 'platter': 0.00125, 'world!': 0.00125, 'very,': 0.00125, 'sad.': 0.0025, 'happened': 0.00125, 'pretty....off': 0.00125, 'putting.': 0.00125, 'evening': 0.00125, 'worst.': 0.00125, "don't'": 0.00125, 'combo': 0.0025, 'ala': 0.025, 'cart?': 0.00125, 'del': 0.035, 'nasty': 0.00375, 'avoided': 0.00125, 'penne': 0.00125, 'vodka': 0.00125, 'excellent!': 0.00125, 'nicest': 0.00125, 'immediately.': 0.00125, 'mistake.': 0.00125, 'shoe': 0.00125, 'leather.': 0.00125, 'voodoo': 0.00125, 'gluten': 0.00125, 'free': 0.0025, 'ago.': 0.00125, 'lady': 0.00125, 'caterpillar': 0.00125, 'arrived,': 0.00125, 'gyros': 0.00125, 'missing.': 0.00125, 'fondue,': 0.00125, 'hungry,': 0.00125, 'stuffed!': 0.00125, 'bisque,': 0.00125, 'bussell': 0.00125, 'sprouts,': 0.00125, 'risotto,': 0.00125, 'filet': 0.00125, 'pepper..and': 0.00125, 'tables.': 0.00125, 'dinners.': 0.00125, 'golden-crispy': 0.00125, 'arrives': 0.00125, 'hands-down': 0.00125, 'metro': 0.00125, 'bisque': 0.0025, 'lukewarm.': 0.00125, 'eyed': 0.00125, 'peas': 0.00125, 'potatoes...': 0.00125, 'unreal!': 0.00125, 'pan': 0.01, 'cakes': 0.00375, 'raving': 0.00125, 'disaster': 0.00125, 'tailored': 0.00125, 'palate': 0.00125, 'six': 0.00125, 'old.': 0.00625, 'hole': 0.0025, 'mexican': 0.00125, 'street': 0.00125, 'tacos,': 0.00125, 'food...': 0.0025, 'dirt.': 0.00125, 'brick': 0.00125, 'oven': 0.0025, 'app!': 0.00125, 'im': 0.10625, 'az': 0.0275, 'call': 0.00875, 'steakhouse': 0.00125, 'properly': 0.00125, 'understand!': 0.00125, '!....the': 0.00125, 'quit': 0.01, 'soooooo': 0.00125, 'wrap': 0.005, 'freaking': 0.00125, 'papers': 0.00125, 'buldogis': 0.00125, 'gourmet': 0.00125, 'cannot': 0.00125, 'unexperienced': 0.00125, 'employees': 0.00125, 'chickens': 0.00125, 'heads': 0.00125, 'cut': 0.00375, 'off.': 0.00125, 'tartar.': 0.00125, 'completely': 0.00125, 'hell': 0.00375, 'stretch': 0.00125, 'imagination.': 0.00125, 'star': 0.01875, '90%': 0.00125, 'lover': 0.00375, 'means.': 0.00125, 'fiancã©': 0.00125, 'middle': 0.00125, 'away.': 0.00125, 'rated.': 0.00375, 'receives': 0.00125, 'appetizers!!!': 0.00125, 'nargile': 0.00125, 'decor': 0.005, 'calligraphy': 0.00125, 'paper.': 0.00125, 'impressive': 0.00125, "hasn't": 0.00125, 'closed': 0.00125, 'down.': 0.00125, 'limited': 0.00125, 'boiled': 0.00125, 'boring.': 0.00125, 'amazing!!': 0.0025, 'cause': 0.01375, 'owned,': 0.00125, 'hella': 0.00125, 'salty.': 0.00125, '-my': 0.00125, 'correct.': 0.00125, 'occasional': 0.00125, 'pats': 0.00125, 'butter...': 0.00125, 'mmmm...!': 0.00125, 'delicious!!': 0.00125, 'due': 0.0025, 'acknowledged,': 0.00125, 'food...and': 0.00125, 'forgetting': 0.00125, 'things.': 0.00125, 'insults,': 0.00125, 'profound': 0.00125, 'deuchebaggery,': 0.00125, 'smoke': 0.00125, 'solidify': 0.00125, 'typical': 0.00125, 'hands': 0.00375, 'restaurant!': 0.00125, 'visit': 0.005, 'hiro': 0.00125, 'delight!': 0.00125, 'positive': 0.00125, 'note,': 0.00125, 'provided': 0.00125, 'ago!': 0.00125, 'bland,': 0.00125, 'overcooked': 0.0025, 'helped': 0.0025, 'loves': 0.0025, 'bone': 0.00125, 'marrow,': 0.00125, 'marrow': 0.00125, 'go!': 0.0025, 'touch.': 0.00125, 'naan': 0.00125, 'pine': 0.0025, 'nut': 0.02375, 'world.': 0.00125, 'cheesecurds': 0.00125, 'drag': 0.00125, 'into': 0.00125, 'sooooo': 0.0025, 'good!!': 0.00125, 'hardly': 0.00125, 'witnessed': 0.00125, 'guests': 0.00125, 'fridays': 0.00125, 'blows.': 0.00125, 'ratio': 0.00375, 'tenders': 0.0025, 'unsatisfying.': 0.00125, 'par,': 0.00125, 'said,': 0.00125, 'mouths': 0.00125, 'bellies': 0.00125, 'pleased.': 0.00125, 'atrocious': 0.00125, 'meal,': 0.00125, 'lunch.': 0.0025, 'gotten': 0.00125, 'door': 0.005, 'services': 0.00125, 'double': 0.00125, 'cheeseburger': 0.00125, 'patty': 0.00125, 'falling': 0.00125, 'apart': 0.00125, '(picture': 0.00125, 'uploaded)': 0.00125, 'yeah,': 0.00125, 'reviewer': 0.00125, '"you': 0.00125, 'again."': 0.00125, 'salad,': 0.00125, 'fabulous': 0.00125, 'vinegrette.': 0.00125, 'strings': 0.00125, 'bottom.': 0.00125, 'disapppointment': 0.00125, 'entrees.': 0.00125, 'here,': 0.005, 'convenient': 0.00125, 'location.': 0.00125, 'terrible,': 0.00125, 'mediocre.': 0.00125, 'glance': 0.00125, 'bakery': 0.00125, 'ambiance,': 0.00125, 'clean,': 0.00125, 'brunch.': 0.00125, 'ripped': 0.00125, 'banana': 0.00125, 'ripped,': 0.00125, 'petrified': 0.00125, 'helpful,': 0.00125, 'boys': 0.00125, 'baby!': 0.00125, 'reheated': 0.00125, 'wedges': 0.00125, 'soggy.': 0.00125, 'main': 0.00375, 'crowd': 0.00125, 'older': 0.0025, 'crowd,': 0.00125, 'mid': 0.0025, '30s': 0.00125, 'no,': 0.00125, 'strangers': 0.00125, 'hair': 0.00125, 'experienced': 0.0025, 'frenchman.': 0.00125, '100': 0.0025, 'home.': 0.00125, 'group': 0.00375, '70+': 0.00125, 'claimed': 0.0025, 'handled': 0.00125, 'beautifully.': 0.00125, '10+': 0.00125, 'stood': 0.00125, 'begin': 0.00125, 'awkwardly': 0.00125, '-drinks': 0.00125, 'point.': 0.00375, 'beauty,': 0.00125, 'dry': 0.005, 'mediterranean': 0.00125, 'love.': 0.00125, 'frustrated.': 0.00125, 'cute.': 0.00125, 'omelets': 0.00125, 'for!': 0.00125, 'summary,': 0.00125, 'largely': 0.00125, 'screams': 0.00125, '"legit"': 0.00125, "book...somethat's": 0.00125, 'jamaican': 0.00125, 'mojitos': 0.00125, 'iced': 0.01125, 'tea.': 0.00125, 'cape': 0.0025, 'cod': 0.00125, 'ravoli,': 0.00125, 'chicken,with': 0.00125, 'cranberry...mmmm!': 0.00125, 'might': 0.00125, 'last.': 0.00125, 'saving': 0.00125, 'this!': 0.00125, "girlfriend's": 0.00125, 'veal': 0.00125, 'lastly,': 0.00125, 'mozzarella': 0.00125, 'sticks,': 0.00125, 'ordered.': 0.00125, 'rice,': 0.00125, '$3': 0.00125, 'paid.': 0.00125, 'refried': 0.00125, 'beans': 0.0025, 'dried': 0.00125, 'crusty': 0.00125, 'furthermore,': 0.00125, 'operation': 0.00125, 'website!': 0.00125, 'leave.': 0.00125, 'join': 0.0025, 'club': 0.00125, 'via': 0.00125, 'email.': 0.00125, 'otto': 0.00375, 'welcome': 0.0025, 'me,': 0.0025, 'fail.': 0.00125, 'amazing...rge': 0.00125, 'fillet': 0.00125, 'relleno': 0.00125, 'plate': 0.0025, 'bruschetta': 0.00125, 'devine.': 0.00125, 'exquisite.': 0.00125, 'next.': 0.00125, '"crumby"': 0.00125, 'decided': 0.00125, 'anyway,': 0.00125, 'fs': 0.00125, 'breakfast/lunch.': 0.00125, 'exceptional': 0.00125, 'reviews.': 0.00125, 'high-quality': 0.00125, 'caesar': 0.00125, 'airport': 0.00125, 'speedy,': 0.00125, '$20,': 0.00125, 'wrong.': 0.00125, 'block': 0.00125, 'ever!': 0.00125, 'reservation.': 0.00125, 'staying': 0.00125, 'mirage.': 0.00125, 'bucks': 0.00125, 'head,': 0.00125, 'pink': 0.00125, 'char': 0.0075, 'outside.': 0.00125, 'shrimp-': 0.00125, 'unwrapped': 0.00125, '(i': 0.00375, '1/2': 0.00125, 'mile': 0.00125, 'brushfire)': 0.00125, 'couple': 0.0025, 'ago': 0.00375, 'great!!!!!!!!!!!!!!': 0.00125, 'waiter.': 0.00125, 'forty': 0.00125, 'five': 0.00125, 'vain.': 0.00125, 'food!': 0.0025, '--': 0.00125, 'touched': 0.00125, 'vegas.....there': 0.00125, 'none.': 0.00125, 'wow...': 0.00125, 'strawberry': 0.00125, 'tea,': 0.00125, 'sangria': 0.00125, 'glass': 0.0025, '$12,': 0.00125, 'ridiculous.': 0.00125, 'sit-down': 0.00125, 'together': 0.00125, 'friends.': 0.00125, 'watch': 0.005, 'food!)': 0.00125, 'forth': 0.00125, '"are': 0.00125, 'helped?"': 0.00125, 'performed.': 0.00125, 'brother': 0.00125, 'law': 0.0025, 'works': 0.00125, 'ate': 0.065, 'same': 0.00125, 'day,': 0.00125, 'be,': 0.00125, 'menus': 0.00125, 'handed': 0.00125, 'listed.': 0.00125, 'negligent': 0.00125, 'unwelcome...': 0.00125, 'suggest': 0.0025, 'elsewhere': 0.00375, 'dessert.': 0.00125, 'level': 0.00125, 'perfect,': 0.00125, 'spice': 0.0025, 'over-whelm': 0.00125, 'soup.': 0.00125, "caballero's": 0.00125, 'week': 0.0025, 'since!': 0.00125, 'dollars': 0.00125, 'horrible.': 0.00125, '40min': 0.00125, 'ordering': 0.00125, 'arriving,': 0.00125, 'busy.': 0.00125, 'highlights': 0.00125, ':': 0.01125, 'nigiri': 0.00125, "owner's": 0.00125, 'people.!': 0.00125, 'eclectic': 0.00125, 'strip,': 0.00125, 'go.': 0.0025, 'mistake': 0.0025, 'was!': 0.00125, 'lil': 0.0025, 'fuzzy': 0.00125, 'nobu,': 0.00125, 'packed!!': 0.00125, 'in-house!': 0.00125, 'you,': 0.00125, "world's": 0.00125, 'worst/annoying': 0.00125, 'drunk': 0.00125, 'people.': 0.0025, 'known': 0.0025, 'excalibur,': 0.00125, 'common': 0.00125, 'sense.': 0.00125, 'experience!': 0.00125, 'eew...': 0.00125, 'complete': 0.0025, 'overhaul.': 0.00125, 'tigerlilly': 0.00125, 'afternoon!': 0.00125, 'miss': 0.00375, 'wish': 0.00125, 'philadelphia!': 0.00125, 'crazy': 0.00125, 'guacamole': 0.00125, 'purã©ed.': 0.00125, 'sucker': 0.00125, 'dry!!.': 0.00125, 'omg,': 0.00125, 'delicioso!': 0.00125, 'hamburger.': 0.00125, 'ha': 0.2575, 'flop.': 0.00125}
    

# conditional probability based on the sentiment
### e.g.: P[“the” | Positive]  = no. of positive documents containing “the” / num of all positive review documents



```python
num_of_positive_docs=0
for pair in train_list:
  if pair[1]==1:
    num_of_positive_docs+=1
print(num_of_positive_docs)    
```

    393
    

## P[word|positive]


```python
positive_prob={}
for pair in positive_count.most_common():
  count=0
  if pair[0].lower() not in stop_words and pair[1]>3:
    for sentence,label in train_list:
      if pair[0].lower() in sentence.lower() and label==1:
        count+=1
      positive_prob[pair[0].lower()]=count/num_of_positive_docs
print(positive_prob)        
```

    {'the': 0.5292620865139949, 'and': 0.39185750636132316, 'was': 0.2544529262086514, 'i': 0.9236641221374046, 'a': 0.9389312977099237, 'is': 0.3256997455470738, 'to': 0.21119592875318066, 'this': 0.13994910941475827, 'great': 0.13231552162849872, 'in': 0.4173027989821883, 'of': 0.10432569974554708, 'good': 0.13740458015267176, 'very': 0.11704834605597965, 'for': 0.09414758269720101, 'food': 0.12468193384223919, 'with': 0.07633587786259542, 'place': 0.10432569974554708, 'my': 0.08396946564885496, 'are': 0.09669211195928754, 'we': 0.17557251908396945, 'it': 0.2900763358778626, 'you': 0.0737913486005089, 'were': 0.06615776081424936, 'on': 0.29770992366412213, 'service': 0.10178117048346055, 'so': 0.20610687022900764, 'they': 0.06361323155216285, 'had': 0.06615776081424936, 'have': 0.06361323155216285, 'our': 0.07124681933842239, 'all': 0.11959287531806616, 'be': 0.17048346055979643, 'that': 0.05089058524173028, 'really': 0.04071246819338423, 'their': 0.035623409669211195, 'not': 0.043256997455470736, 'just': 0.04071246819338423, 'time': 0.05089058524173028, 'as': 0.40966921119592875, 'will': 0.035623409669211195, 'nice': 0.043256997455470736, 'also': 0.035623409669211195, 'friendly': 0.05089058524173028, 'first': 0.03307888040712468, 'here': 0.08396946564885496, 'but': 0.043256997455470736, 'at': 0.3333333333333333, 'back': 0.04071246819338423, 'best': 0.030534351145038167, 'an': 0.5267175572519084, 'good.': 0.027989821882951654, 'loved': 0.02544529262086514, 'he': 0.6132315521628499, 'by': 0.027989821882951654, 'restaurant': 0.030534351145038167, 'love': 0.06361323155216285, 'go': 0.1984732824427481, 'some': 0.05089058524173028, '-': 0.04834605597964377, 'place.': 0.022900763358778626, 'one': 0.04834605597964377, 'chicken': 0.020356234096692113, 'what': 0.020356234096692113, 'server': 0.022900763358778626, 'staff': 0.03307888040712468, 'fresh': 0.02544529262086514, 'vegas': 0.035623409669211195, 'only': 0.020356234096692113, 'like': 0.027989821882951654, 'definitely': 0.020356234096692113, 'pretty': 0.020356234096692113, 'service.': 0.020356234096692113, 'always': 0.022900763358778626, 'been': 0.015267175572519083, 'us': 0.17557251908396945, "i'm": 0.017811704834605598, 'came': 0.017811704834605598, 'when': 0.020356234096692113, 'perfect': 0.03307888040712468, 'fantastic': 0.027989821882951654, 'happy': 0.017811704834605598, 'made': 0.022900763358778626, 'get': 0.022900763358778626, 'every': 0.03307888040712468, 'has': 0.015267175572519083, 'even': 0.022900763358778626, 'amazing.': 0.017811704834605598, 'awesome': 0.022900763358778626, 'food.': 0.015267175572519083, 'out': 0.058524173027989825, 'or': 0.20865139949109415, 'which': 0.015267175572519083, 'if': 0.027989821882951654, 'food,': 0.015267175572519083, 'could': 0.017811704834605598, 'come': 0.020356234096692113, 'would': 0.015267175572519083, 'wonderful': 0.015267175572519083, 'say': 0.017811704834605598, 'going': 0.01272264631043257, 'order': 0.02544529262086514, 'went': 0.015267175572519083, 'your': 0.01272264631043257, "it's": 0.01272264631043257, 'delicious!': 0.015267175572519083, 'great.': 0.01272264631043257, 'thing': 0.03307888040712468, 'everything': 0.015267175572519083, 'menu': 0.020356234096692113, 'staff.': 0.01272264631043257, 'bacon': 0.01272264631043257, 'taste': 0.017811704834605598, 'worth': 0.01272264631043257, 'recommend': 0.022900763358778626, 'good,': 0.010178117048346057, 'pizza': 0.017811704834605598, 'great,': 0.01272264631043257, 'can': 0.03816793893129771, 'here.': 0.020356234096692113, 'delicious.': 0.01272264631043257, 'experience.': 0.01272264631043257, 'sauce': 0.01272264631043257, '5': 0.01272264631043257, 'about': 0.01272264631043257, 'still': 0.01272264631043257, 'tried': 0.01272264631043257, 'vegas.': 0.01272264631043257, 'bread': 0.01272264631043257, 'than': 0.01272264631043257, 'more': 0.01272264631043257, '&': 0.007633587786259542, 'excellent': 0.017811704834605598, 'super': 0.010178117048346057, 'little': 0.010178117048346057, 'fantastic.': 0.010178117048346057, 'any': 0.022900763358778626, 'up': 0.043256997455470736, 'buffet': 0.01272264631043257, 'spot.': 0.010178117048346057, 'inside': 0.015267175572519083, 'quite': 0.01272264631043257, 'breakfast': 0.015267175572519083, 'check': 0.015267175572519083, 'did': 0.022900763358778626, 'there': 0.010178117048346057, 'beer': 0.01272264631043257, 'both': 0.010178117048346057, 'want': 0.010178117048346057, 'night': 0.015267175572519083, 'from': 0.010178117048346057, "didn't": 0.010178117048346057, 'waitress': 0.010178117048346057, 'friendly.': 0.010178117048346057, 'absolutely': 0.010178117048346057, 'selection': 0.020356234096692113, 'salad': 0.020356234096692113, 'how': 0.01272264631043257, 'potato': 0.015267175572519083, "i've": 0.010178117048346057, 'ever': 0.061068702290076333, 'amazing': 0.043256997455470736, 'family': 0.010178117048346057, 'never': 0.01272264631043257, 'prices': 0.020356234096692113, 'steak': 0.022900763358778626, 'once': 0.010178117048346057, 'spicy': 0.010178117048346057, 'quality': 0.01272264631043257, 'it!': 0.010178117048346057, 'sweet': 0.010178117048346057, 'that.': 0.010178117048346057, "can't": 0.010178117048346057}
    


```python
num_of_negative_docs=0
for pair in train_list:
  if pair[1]==0:
    num_of_negative_docs+=1
print(num_of_negative_docs)
```

    407
    

## P[word|negative]


```python
negative_prob={}
for pair in negative_count.most_common():
  count=0
  if pair[0].lower() not in stop_words and pair[1]>3:
    for sentence,label in train_list:
      if pair[0].lower() in sentence.lower() and label==1:
        count+=1
      negative_prob[pair[0].lower()]=count/num_of_negative_docs
print(negative_prob)    
```

    {'the': 0.5110565110565111, 'i': 0.8918918918918919, 'was': 0.2457002457002457, 'and': 0.3783783783783784, 'to': 0.20393120393120392, 'a': 0.9066339066339066, 'not': 0.04176904176904177, 'it': 0.2800982800982801, 'for': 0.09090909090909091, 'is': 0.3144963144963145, 'of': 0.10073710073710074, 'this': 0.13513513513513514, 'food': 0.12039312039312039, 'we': 0.16953316953316952, 'be': 0.16461916461916462, 'in': 0.40294840294840295, 'at': 0.32186732186732187, 'that': 0.04914004914004914, 'but': 0.04176904176904177, 'place': 0.10073710073710074, 'my': 0.08108108108108109, 'had': 0.06388206388206388, 'so': 0.19901719901719903, 'like': 0.02702702702702703, 'have': 0.06142506142506143, 'they': 0.06142506142506143, 'service': 0.09828009828009827, 'very': 0.11302211302211303, 'were': 0.06388206388206388, 'with': 0.07371007371007371, 'you': 0.07125307125307126, 'never': 0.012285012285012284, "don't": 0.002457002457002457, 'go': 0.19164619164619165, 'if': 0.02702702702702703, 'are': 0.09336609336609336, 'will': 0.0343980343980344, 'there': 0.009828009828009828, 'on': 0.28746928746928746, 'back': 0.03931203931203931, 'no': 0.09090909090909091, 'just': 0.03931203931203931, 'would': 0.014742014742014743, 'here': 0.08108108108108109, 'our': 0.0687960687960688, 'as': 0.3955773955773956, 'got': 0.004914004914004914, 'your': 0.012285012285012284, 'from': 0.009828009828009828, "won't": 0.007371007371007371, 'did': 0.022113022113022112, 'only': 0.019656019656019656, 'been': 0.014742014742014743, 'out': 0.056511056511056514, 'eat': 0.16216216216216217, 'up': 0.04176904176904177, 'time': 0.04914004914004914, 'good': 0.13267813267813267, 'ever': 0.05896805896805897, 'minutes': 0.0, 'what': 0.019656019656019656, 'much': 0.002457002457002457, 'bad': 0.0, 'get': 0.022113022113022112, 'more': 0.012285012285012284, 'their': 0.0343980343980344, 'food.': 0.014742014742014743, 'an': 0.5085995085995086, 'going': 0.012285012285012284, 'really': 0.03931203931203931, 'it.': 0.0171990171990172, 'one': 0.04668304668304668, 'or': 0.20147420147420148, "wasn't": 0.0, 'being': 0.002457002457002457, "i've": 0.009828009828009828, 'again.': 0.004914004914004914, 'when': 0.019656019656019656, 'which': 0.014742014742014743, 'came': 0.0171990171990172, "i'm": 0.0171990171990172, 'back.': 0.007371007371007371, 'probably': 0.0, 'too': 0.004914004914004914, 'how': 0.012285012285012284, 'better': 0.009828009828009828, 'than': 0.012285012285012284, 'all': 0.11547911547911548, 'by': 0.02702702702702703, '-': 0.04668304668304668, 'about': 0.012285012285012284, 'other': 0.009828009828009828, 'worst': 0.0, 'definitely': 0.019656019656019656, 'after': 0.007371007371007371, "didn't": 0.009828009828009828, 'quality': 0.012285012285012284, 'us': 0.16953316953316952, 'me': 0.23832923832923833, 'think': 0.007371007371007371, 'know': 0.004914004914004914, 'coming': 0.004914004914004914, 'could': 0.0171990171990172, 'some': 0.04914004914004914, 'pretty': 0.019656019656019656, 'wait': 0.022113022113022112, 'even': 0.022113022113022112, 'feel': 0.012285012285012284, 'any': 0.022113022113022112, 'waited': 0.0, 'because': 0.007371007371007371, 'bit': 0.007371007371007371, 'chicken': 0.019656019656019656, 'slow': 0.0, 'over': 0.0171990171990172, 'do': 0.02702702702702703, 'can': 0.036855036855036855, "it's": 0.012285012285012284, 'took': 0.0, 'give': 0.004914004914004914, 'down': 0.009828009828009828, 'then': 0.007371007371007371, 'said': 0.002457002457002457, 'she': 0.014742014742014743, 'made': 0.022113022113022112, 'way': 0.044226044226044224, 'good.': 0.02702702702702703, 'her': 0.09336609336609336, '&': 0.007371007371007371, 'times': 0.007371007371007371, 'went': 0.014742014742014743, 'he': 0.5921375921375921, 'another': 0.002457002457002457, 'poor': 0.0, 'around': 0.002457002457002457, 'take': 0.002457002457002457, 'sushi': 0.012285012285012284, 'meat': 0.004914004914004914, 'want': 0.009828009828009828, 'service.': 0.019656019656019656, 'barely': 0.0, 'bad.': 0.0, 'should': 0.0, 'fries': 0.009828009828009828, 'hard': 0.002457002457002457, 'getting': 0.002457002457002457, 'felt': 0.002457002457002457, 'little': 0.009828009828009828, "can't": 0.009828009828009828, 'best.': 0.0, 'two': 0.007371007371007371, 'worth': 0.012285012285012284, 'still': 0.012285012285012284, 'asked': 0.0, '30': 0.0, 'has': 0.014742014742014743, 'enough': 0.009828009828009828, 'waste': 0.0, 'experience': 0.02702702702702703, 'next': 0.009828009828009828, 'all,': 0.007371007371007371, 'off': 0.007371007371007371, 'here!': 0.004914004914004914, 'zero': 0.0, 'mediocre': 0.0, 'terrible': 0.0, 'cold.': 0.0, 'before': 0.0, 'night': 0.014742014742014743, 'chips': 0.004914004914004914, 'kept': 0.002457002457002457, 'many': 0.004914004914004914, 'bland': 0.0, 'sure': 0.007371007371007371, 'up.': 0.007371007371007371, 'disappointed': 0.009828009828009828, 'now': 0.014742014742014743, 'pizza': 0.0171990171990172, 'restaurant.': 0.004914004914004914, 'also': 0.0343980343980344, 'ordered': 0.009828009828009828, 'them': 0.007371007371007371, 'eating': 0.002457002457002457, 'servers': 0.002457002457002457, 'server': 0.022113022113022112, 'anytime': 0.0, 'dining': 0.007371007371007371, 'who': 0.004914004914004914, 'money': 0.0, 'waiting': 0.0, 'bring': 0.002457002457002457, 'few': 0.004914004914004914, 'hour': 0.002457002457002457, '2': 0.012285012285012284, 'dishes': 0.009828009828009828, 'make': 0.007371007371007371, "i'll": 0.004914004914004914, 'say': 0.0171990171990172, 'there.': 0.0, 'long': 0.0, 'tasted': 0.002457002457002457, 'enjoy': 0.009828009828009828, 'business': 0.0, 'tasteless.': 0.0, 'live': 0.0}
    


```python
len(train_list)
```




    800




```python
prob_of_pos=num_of_positive_docs/len(train_list)
prob_of_neg=num_of_negative_docs/len(train_list)

print(prob_of_pos)
print(prob_of_neg)
```

    0.49125
    0.50875
    
 
# Function to Predict the sentiment


```python
def predict(sentence):
  pos_prob=1
  neg_prob=1
  for word in sentence.split(' '):
    if word.lower() in prob_all_docs.keys() and word.lower() not in stop_words:
      # print(word) 
      if word.lower() not in positive_prob:
        pos_prob=pos_prob*0
      else:
        pos_prob=pos_prob*positive_prob[word.lower()]  
       
      if word.lower() not in negative_prob:
        neg_prob=neg_prob*0
      else:
        neg_prob=neg_prob*negative_prob[word.lower()] 
        # print(pos_prob) 
        # print(neg_prob)
  pos_prob=pos_prob*prob_of_pos
  neg_prob=neg_prob*prob_of_neg
  if pos_prob>neg_prob:
    return 1,pos_prob
  else:
    return 0,neg_prob     



```

# Predicting the sentiment using the test data


```python
test_list=test_set.values.tolist()
count=0
for sentence in test_list:
  # print(sentence[0])
  pred,prob=predict(sentence[0])
  if pred==sentence[1]:
    count+=1
  print(pred,prob)
print('accuracy = ',count/len(test_list))    

```

    0 0.0
    1 0.013514169725928948
    1 2.696663754021951e-07
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 3.1501445409345893e-10
    0 0.0
    1 0.010000000000000002
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 5.908618072449519e-05
    1 0.00012917626555152795
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 1.8467039857791e-08
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 0.001784351145038168
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 1.4989092033586949e-05
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 1.314695693170586e-06
    0 0.00010465653278921092
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 4.755695719133267e-09
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    0 0.25875000000000004
    1 0.00023794262183633434
    1 2.99500199815623e-05
    0 0.0
    0 0.0
    0 0.0
    1 6.538551713102034e-05
    0 0.0
    1 4.1417827190025365e-10
    0 0.0
    0 0.00011948366781355388
    0 0.0
    1 5.244449624147777e-06
    0 0.0
    0 0.0
    0 0.0
    0 0.0
    1 0.002888851029606898
    accuracy =  0.54
    

# Conducting 5 fold crossvalidation


```python
val_list=val_set.values.tolist()

val1=[]
val2=[]
val3=[]
val4=[]
val5=[]

count1=0

acc_list=[]
acc_list1=[]

for pair in val_list:
  if count1<20:
    val1.append(pair)
  elif count1<40 and count1>=20:
    val2.append(pair)
  elif count1<60 and count1>=40:
    val3.append(pair)  
  elif count1<80 and count1>=60:
    val4.append(pair)  
  elif count1<100 and count1>=80:
    val5.append(pair)
  count1+=1            
# print(val1)
# print(val2)

crossval=[val1,val2,val3,val4,val5]

train_set=[]

for i in range(0,5):
  
  test=crossval[i]
  for j in range(0,5):
    if j!=i:
      for pair in crossval[j]:
        train_set.append(pair)

    # print('train set',train_set)
  train_list=train_set
  for i in range(len(train_set)):
    if (train_list[i][1] == 1):
      for word in train_list[i][0].split(" "):
        positive_count[word.lower()] += 1
        total_count[word.lower()] += 1
    else:
      for word in train_list[i][0].split(" "):
          negative_count[word.lower()] += 1
          total_count[word.lower()] += 1
    # print(positive_count.most_common())           

    total_words=0
    for pair in total_count.most_common():
      total_words+=pair[1]
    # print(total_words)  

    prob_all_docs={}
# stop_words=['','the','a','and','of','is','to','this','was','in','that','it','for','as','with','are','on','This','i']
    stop_words=['']
    for pair in total_count.most_common():
      count2=0
      if pair[0].lower() not in stop_words:
        for sentence in train_list:
          if pair[0].lower() in sentence[0].lower():
            count2=count2+1
      prob_all_docs[pair[0]]=count2/len(train_list)
    # print('prob all docs',prob_all_docs)
    
    num_of_positive_docs=0
    # print('aa')
    for pair in train_list:
      if pair[1]==1:
        # print('aa')
        num_of_positive_docs+=1

    positive_prob={}
    for pair in positive_count.most_common():
      count=0
      if pair[0].lower() not in stop_words and pair[1]>3:
        for sentence,label in train_list:
          if pair[0].lower() in sentence.lower() and label==1:
            count+=1
          positive_prob[pair[0].lower()]=count/num_of_positive_docs

    num_of_negative_docs=0
    for pair in train_list:
      if pair[1]==0:
        num_of_negative_docs+=1

    negative_prob={}
    for pair in negative_count.most_common():
      count=0
      if pair[0].lower() not in stop_words and pair[1]>3:
        for sentence,label in train_list:
          if pair[0].lower() in sentence.lower() and label==1:
            count+=1
          negative_prob[pair[0].lower()]=count/num_of_negative_docs

    prob_of_pos=num_of_positive_docs/len(train_list)
    prob_of_neg=num_of_negative_docs/len(train_list)
    
    smoothened_positive_prob={}
    for pair in positive_count.most_common():
      count=0
      if pair[0].lower() not in stop_words and pair[1]>3:
        for sentence,label in train_list:
          if pair[0].lower() in sentence.lower() and label==1:
            count+=1
      smoothened_positive_prob[pair[0].lower()]=(count+1)/(num_of_positive_docs+2)

    smoothened_negative_prob={}
    for pair in negative_count.most_common():
      count=0
      if pair[0].lower() not in stop_words and pair[1]>3:
        for sentence,label in train_list:
          if pair[0].lower() in sentence.lower() and label==1:
            count+=1
      smoothened_negative_prob[pair[0].lower()]=(count+1)/(num_of_negative_docs+2)



    count=0
    count1=0
    for pair in test:
      pred,prob=predict(pair[0])
      pred1,prob1=smoothened_predict(pair[0])
      if pred==pair[1]:
        count+=1
      if pred1==pair[1]:
        count1=count1+1
    
    
    accuracy =count/len(val_list)
    accuracy1=count1/len(val_list)
    # print(accuracy)
    acc_list.append(accuracy)
    acc_list1.append(accuracy1)

avgacc=sum(acc_list)/5
avgacc1=sum(acc_list1)/5
print("average accuracy of 5 fold cross validation is: ",avgacc)
print('average accuracy of 5 fold cross validation using smoothening is: ',avgacc1)
      

       
  


```

    average accuracy of 5 fold cross validation is:  41.46799999999937
    average accuracy of 5 fold cross validation using smoothening is:  44.94599999999919
    


```python
smoothened_negative_prob={}
for pair in negative_count.most_common():
  count=0
  if pair[0].lower() not in stop_words and pair[1]>3:
    for sentence,label in train_list:
      if pair[0].lower() in sentence.lower() and label==1:
        count+=1
      smoothened_negative_prob[pair[0].lower()]=(count+1)/(num_of_negative_docs+2)
print(smoothened_negative_prob)  
```

    {'the': 0.511002444987775, 'i': 0.8899755501222494, 'was': 0.2469437652811736, 'and': 0.37897310513447435, 'to': 0.20537897310513448, 'a': 0.9046454767726161, 'not': 0.044009779951100246, 'it': 0.28117359413202936, 'for': 0.09290953545232274, 'is': 0.3154034229828851, 'of': 0.10268948655256724, 'this': 0.13691931540342298, 'food': 0.12224938875305623, 'we': 0.17114914425427874, 'be': 0.16625916870415647, 'in': 0.4034229828850856, 'at': 0.32273838630806845, 'that': 0.05134474327628362, 'but': 0.044009779951100246, 'place': 0.10268948655256724, 'my': 0.08312958435207823, 'had': 0.06601466992665037, 'so': 0.20048899755501223, 'like': 0.029339853300733496, 'have': 0.06356968215158924, 'they': 0.06356968215158924, 'service': 0.10024449877750612, 'very': 0.11491442542787286, 'were': 0.06601466992665037, 'with': 0.07579462102689487, 'you': 0.07334963325183375, 'never': 0.014669926650366748, "don't": 0.004889975550122249, 'go': 0.19315403422982885, 'if': 0.029339853300733496, 'are': 0.09535452322738386, 'will': 0.03667481662591687, 'there': 0.012224938875305624, 'on': 0.2885085574572127, 'back': 0.04156479217603912, 'no': 0.09290953545232274, 'just': 0.04156479217603912, 'would': 0.017114914425427872, 'here': 0.08312958435207823, 'our': 0.07090464547677261, 'as': 0.3960880195599022, 'got': 0.007334963325183374, 'your': 0.014669926650366748, 'from': 0.012224938875305624, "won't": 0.009779951100244499, 'did': 0.02444987775061125, 'only': 0.022004889975550123, 'been': 0.017114914425427872, 'out': 0.05867970660146699, 'eat': 0.16381418092909536, 'up': 0.044009779951100246, 'time': 0.05134474327628362, 'good': 0.13447432762836187, 'ever': 0.061124694376528114, 'minutes': 0.0024449877750611247, 'what': 0.022004889975550123, 'much': 0.004889975550122249, 'bad': 0.0024449877750611247, 'get': 0.02444987775061125, 'more': 0.014669926650366748, 'their': 0.03667481662591687, 'food.': 0.017114914425427872, 'an': 0.508557457212714, 'going': 0.014669926650366748, 'really': 0.04156479217603912, 'it.': 0.019559902200488997, 'one': 0.0488997555012225, 'or': 0.20293398533007334, "wasn't": 0.0024449877750611247, 'being': 0.004889975550122249, "i've": 0.012224938875305624, 'again.': 0.007334963325183374, 'when': 0.022004889975550123, 'which': 0.017114914425427872, 'came': 0.019559902200488997, "i'm": 0.019559902200488997, 'back.': 0.009779951100244499, 'probably': 0.0024449877750611247, 'too': 0.007334963325183374, 'how': 0.014669926650366748, 'better': 0.012224938875305624, 'than': 0.014669926650366748, 'all': 0.11735941320293398, 'by': 0.029339853300733496, '-': 0.0488997555012225, 'about': 0.014669926650366748, 'other': 0.012224938875305624, 'worst': 0.0024449877750611247, 'definitely': 0.022004889975550123, 'after': 0.009779951100244499, "didn't": 0.012224938875305624, 'quality': 0.014669926650366748, 'us': 0.17114914425427874, 'me': 0.2396088019559902, 'think': 0.009779951100244499, 'know': 0.007334963325183374, 'coming': 0.007334963325183374, 'could': 0.019559902200488997, 'some': 0.05134474327628362, 'pretty': 0.022004889975550123, 'wait': 0.02444987775061125, 'even': 0.02444987775061125, 'feel': 0.014669926650366748, 'any': 0.02444987775061125, 'waited': 0.0024449877750611247, 'because': 0.009779951100244499, 'bit': 0.009779951100244499, 'chicken': 0.022004889975550123, 'slow': 0.0024449877750611247, 'over': 0.019559902200488997, 'do': 0.029339853300733496, 'can': 0.039119804400977995, "it's": 0.014669926650366748, 'took': 0.0024449877750611247, 'give': 0.007334963325183374, 'down': 0.012224938875305624, 'then': 0.009779951100244499, 'said': 0.004889975550122249, 'she': 0.017114914425427872, 'made': 0.02444987775061125, 'way': 0.04645476772616137, 'good.': 0.029339853300733496, 'her': 0.09535452322738386, '&': 0.009779951100244499, 'times': 0.009779951100244499, 'went': 0.017114914425427872, 'he': 0.5916870415647921, 'another': 0.004889975550122249, 'poor': 0.0024449877750611247, 'around': 0.004889975550122249, 'take': 0.004889975550122249, 'sushi': 0.014669926650366748, 'meat': 0.007334963325183374, 'want': 0.012224938875305624, 'service.': 0.022004889975550123, 'barely': 0.0024449877750611247, 'bad.': 0.0024449877750611247, 'should': 0.0024449877750611247, 'fries': 0.012224938875305624, 'hard': 0.004889975550122249, 'getting': 0.004889975550122249, 'felt': 0.004889975550122249, 'little': 0.012224938875305624, "can't": 0.012224938875305624, 'best.': 0.0024449877750611247, 'two': 0.009779951100244499, 'worth': 0.014669926650366748, 'still': 0.014669926650366748, 'asked': 0.0024449877750611247, '30': 0.0024449877750611247, 'has': 0.017114914425427872, 'enough': 0.012224938875305624, 'waste': 0.0024449877750611247, 'experience': 0.029339853300733496, 'next': 0.012224938875305624, 'all,': 0.009779951100244499, 'off': 0.009779951100244499, 'here!': 0.007334963325183374, 'zero': 0.0024449877750611247, 'mediocre': 0.0024449877750611247, 'terrible': 0.0024449877750611247, 'cold.': 0.0024449877750611247, 'before': 0.0024449877750611247, 'night': 0.017114914425427872, 'chips': 0.007334963325183374, 'kept': 0.004889975550122249, 'many': 0.007334963325183374, 'bland': 0.0024449877750611247, 'sure': 0.009779951100244499, 'up.': 0.009779951100244499, 'disappointed': 0.012224938875305624, 'now': 0.017114914425427872, 'pizza': 0.019559902200488997, 'restaurant.': 0.007334963325183374, 'also': 0.03667481662591687, 'ordered': 0.012224938875305624, 'them': 0.009779951100244499, 'eating': 0.004889975550122249, 'servers': 0.004889975550122249, 'server': 0.02444987775061125, 'anytime': 0.0024449877750611247, 'dining': 0.009779951100244499, 'who': 0.007334963325183374, 'money': 0.0024449877750611247, 'waiting': 0.0024449877750611247, 'bring': 0.004889975550122249, 'few': 0.007334963325183374, 'hour': 0.004889975550122249, '2': 0.014669926650366748, 'dishes': 0.012224938875305624, 'make': 0.009779951100244499, "i'll": 0.007334963325183374, 'say': 0.019559902200488997, 'there.': 0.0024449877750611247, 'long': 0.0024449877750611247, 'tasted': 0.004889975550122249, 'enjoy': 0.012224938875305624, 'business': 0.0024449877750611247, 'tasteless.': 0.0024449877750611247, 'live': 0.0024449877750611247}
    


```python
smoothened_positive_prob={}
for pair in positive_count.most_common():
  count=0
  if pair[0].lower() not in stop_words and pair[1]>3:
    for sentence,label in train_list:
      if pair[0].lower() in sentence.lower() and label==1:
        count+=1
      smoothened_positive_prob[pair[0].lower()]=(count+1)/(num_of_positive_docs+2)
print(smoothened_positive_prob)  
```

    {'the': 0.529113924050633, 'and': 0.3924050632911392, 'was': 0.25569620253164554, 'i': 0.9215189873417722, 'a': 0.9367088607594937, 'is': 0.3265822784810127, 'to': 0.21265822784810126, 'this': 0.14177215189873418, 'great': 0.1341772151898734, 'in': 0.4177215189873418, 'of': 0.10632911392405063, 'good': 0.13924050632911392, 'very': 0.1189873417721519, 'for': 0.09620253164556962, 'food': 0.12658227848101267, 'with': 0.07848101265822785, 'place': 0.10632911392405063, 'my': 0.08607594936708861, 'are': 0.09873417721518987, 'we': 0.17721518987341772, 'it': 0.2911392405063291, 'you': 0.0759493670886076, 'were': 0.06835443037974684, 'on': 0.29873417721518986, 'service': 0.10379746835443038, 'so': 0.20759493670886076, 'they': 0.06582278481012659, 'had': 0.06835443037974684, 'have': 0.06582278481012659, 'our': 0.07341772151898734, 'all': 0.12151898734177215, 'be': 0.17215189873417722, 'that': 0.053164556962025315, 'really': 0.043037974683544304, 'their': 0.0379746835443038, 'not': 0.04556962025316456, 'just': 0.043037974683544304, 'time': 0.053164556962025315, 'as': 0.41012658227848103, 'will': 0.0379746835443038, 'nice': 0.04556962025316456, 'also': 0.0379746835443038, 'friendly': 0.053164556962025315, 'first': 0.035443037974683546, 'here': 0.08607594936708861, 'but': 0.04556962025316456, 'at': 0.3341772151898734, 'back': 0.043037974683544304, 'best': 0.03291139240506329, 'an': 0.5265822784810127, 'good.': 0.030379746835443037, 'loved': 0.027848101265822784, 'he': 0.6126582278481013, 'by': 0.030379746835443037, 'restaurant': 0.03291139240506329, 'love': 0.06582278481012659, 'go': 0.2, 'some': 0.053164556962025315, '-': 0.05063291139240506, 'place.': 0.02531645569620253, 'one': 0.05063291139240506, 'chicken': 0.02278481012658228, 'what': 0.02278481012658228, 'server': 0.02531645569620253, 'staff': 0.035443037974683546, 'fresh': 0.027848101265822784, 'vegas': 0.0379746835443038, 'only': 0.02278481012658228, 'like': 0.030379746835443037, 'definitely': 0.02278481012658228, 'pretty': 0.02278481012658228, 'service.': 0.02278481012658228, 'always': 0.02531645569620253, 'been': 0.017721518987341773, 'us': 0.17721518987341772, "i'm": 0.020253164556962026, 'came': 0.020253164556962026, 'when': 0.02278481012658228, 'perfect': 0.035443037974683546, 'fantastic': 0.030379746835443037, 'happy': 0.020253164556962026, 'made': 0.02531645569620253, 'get': 0.02531645569620253, 'every': 0.035443037974683546, 'has': 0.017721518987341773, 'even': 0.02531645569620253, 'amazing.': 0.020253164556962026, 'awesome': 0.02531645569620253, 'food.': 0.017721518987341773, 'out': 0.060759493670886074, 'or': 0.21012658227848102, 'which': 0.017721518987341773, 'if': 0.030379746835443037, 'food,': 0.017721518987341773, 'could': 0.020253164556962026, 'come': 0.02278481012658228, 'would': 0.017721518987341773, 'wonderful': 0.017721518987341773, 'say': 0.020253164556962026, 'going': 0.015189873417721518, 'order': 0.027848101265822784, 'went': 0.017721518987341773, 'your': 0.015189873417721518, "it's": 0.015189873417721518, 'delicious!': 0.017721518987341773, 'great.': 0.015189873417721518, 'thing': 0.035443037974683546, 'everything': 0.017721518987341773, 'menu': 0.02278481012658228, 'staff.': 0.015189873417721518, 'bacon': 0.015189873417721518, 'taste': 0.020253164556962026, 'worth': 0.015189873417721518, 'recommend': 0.02531645569620253, 'good,': 0.012658227848101266, 'pizza': 0.020253164556962026, 'great,': 0.015189873417721518, 'can': 0.04050632911392405, 'here.': 0.02278481012658228, 'delicious.': 0.015189873417721518, 'experience.': 0.015189873417721518, 'sauce': 0.015189873417721518, '5': 0.015189873417721518, 'about': 0.015189873417721518, 'still': 0.015189873417721518, 'tried': 0.015189873417721518, 'vegas.': 0.015189873417721518, 'bread': 0.015189873417721518, 'than': 0.015189873417721518, 'more': 0.015189873417721518, '&': 0.010126582278481013, 'excellent': 0.020253164556962026, 'super': 0.012658227848101266, 'little': 0.012658227848101266, 'fantastic.': 0.012658227848101266, 'any': 0.02531645569620253, 'up': 0.04556962025316456, 'buffet': 0.015189873417721518, 'spot.': 0.012658227848101266, 'inside': 0.017721518987341773, 'quite': 0.015189873417721518, 'breakfast': 0.017721518987341773, 'check': 0.017721518987341773, 'did': 0.02531645569620253, 'there': 0.012658227848101266, 'beer': 0.015189873417721518, 'both': 0.012658227848101266, 'want': 0.012658227848101266, 'night': 0.017721518987341773, 'from': 0.012658227848101266, "didn't": 0.012658227848101266, 'waitress': 0.012658227848101266, 'friendly.': 0.012658227848101266, 'absolutely': 0.012658227848101266, 'selection': 0.02278481012658228, 'salad': 0.02278481012658228, 'how': 0.015189873417721518, 'potato': 0.017721518987341773, "i've": 0.012658227848101266, 'ever': 0.06329113924050633, 'amazing': 0.04556962025316456, 'family': 0.012658227848101266, 'never': 0.015189873417721518, 'prices': 0.02278481012658228, 'steak': 0.02531645569620253, 'once': 0.012658227848101266, 'spicy': 0.012658227848101266, 'quality': 0.015189873417721518, 'it!': 0.012658227848101266, 'sweet': 0.012658227848101266, 'that.': 0.012658227848101266, "can't": 0.012658227848101266}
    

# Function for predicting the sentiment using laplacian smoothening


```python
def smoothened_predict(sentence):
  pos_prob=1
  neg_prob=1
  for word in sentence.split(' '):
    if word.lower() in prob_all_docs.keys() and word.lower() not in stop_words:
      # print(word) 
      if word.lower() not in positive_prob:
        pos_prob=pos_prob*(1/(num_of_positive_docs+2))
      else:
        pos_prob=pos_prob*smoothened_positive_prob[word.lower()]  
       
      if word.lower() not in negative_prob:
        neg_prob=neg_prob*(1/(num_of_negative_docs+2))
      else:
        neg_prob=neg_prob*smoothened_negative_prob[word.lower()] 
        # print(pos_prob) 
        # print(neg_prob)
  pos_prob=pos_prob*prob_of_pos
  neg_prob=neg_prob*prob_of_neg
  if pos_prob>neg_prob:
    return 1,pos_prob
  else:
    return 0,neg_prob    
```

# predicting sentiment of test data using smoothened(laplacian) predictor


```python

test_list=test_set.values.tolist()
print(test_list)
count=0
for sentence in test_list:
  # print(sentence[0])
  pred,prob=smoothened_predict(sentence[0])
  if pred==sentence[1]:
    count+=1
  print(pred,prob)
print('accuracy = ',count/len(test_list)) 
```

    [['Great place to relax and have an awesome burger and beer.', 1], ['I was VERY disappointed!!', 0], ['The food was outstanding and the prices were very reasonable.', 1], ["Each day of the week they have a different deal and it's all so delicious!", 1], ['My husband and I ate lunch here and were very disappointed with the food and service.', 0], ['After I pulled up my car I waited for another 15 minutes before being acknowledged.', 0], ['I have been here several times in the past, and the experience has always been great.', 1], ['I was disgusted because I was pretty sure that was human hair.', 0], ['Ordered an appetizer and took 40 minutes and then the pizza another 10 minutes.', 0], ['Awful service.', 0], ['Service sucks.', 0], ['When I received my Pita it was huge it did have a lot of meat in it so thumbs up there.', 1], ['If you love authentic Mexican food and want a whole bunch of interesting, yet delicious meats to choose from, you need to try this place.', 1], ['I think not again', 0], ["It's a great place and I highly recommend it.", 1], ['The Macarons here are insanely good.', 1], ["That's right....the red velvet cake.....ohhh this stuff is so good.", 1], ['They also have the best cheese crisp in town.', 1], ['I found a six inch long piece of wire in my salsa.', 0], ['I ordered the Lemon raspberry ice cocktail which was also incredible.', 1], ['I had heard good things about this place, but it exceeding every hope I could have dreamed of.', 1], ['Wow very spicy but delicious.', 1], ["Best fish I've ever had in my life!", 1], ['What I really like there is the crepe station.', 1], ['walked in and the place smelled like an old grease trap and only 2 others there eating.', 0], ['Would come back again if I had a sushi craving while in Vegas.', 1], ['An absolute must visit!', 1], ['Weird vibe from owners.', 0], ['Worst martini ever!', 0], ["Horrible - don't waste your time and money.", 0], ['Not my thing.', 0], ['Good beer & drink selection and good food selection.', 1], ['The burger... I got the "Gold Standard" a $17 burger and was kind of disappointed.', 0], ["I've lived here since 1979 and this was the first (and last) time I've stepped foot into this place.", 0], ['say bye bye to your tip lady!', 0], ['The buffet is small and all the food they offered was BLAND.', 0], ['The cocktails are all handmade and delicious.', 1], ['I swung in to give them a try but was deeply disappointed.', 0], ['The atmosphere here is fun.', 1], ['The staff are now not as friendly, the wait times for being served are horrible, no one even says hi for the first 10 minutes.', 0], ['Great place fo take out or eat in.', 1], ["Honeslty it didn't taste THAT fresh.)", 0], ["We won't be returning.", 0], ['Unfortunately, we must have hit the bakery on leftover day because everything we ordered was STALE.', 0], ['Their rotating beers on tap is also a highlight of this place.', 1], ['Tonight I had the Elk Filet special...and it sucked.', 0], ["It's like a really sexy party in your mouth, where you're outrageously flirting with the hottest person at the party.", 1], ['not even a "hello, we will be right with you."', 0], ['Highly unprofessional and rude to a loyal patron!', 0], ["Plus, it's only 8 bucks.", 1], ['This place deserves no stars.', 0], ['Only Pros : Large seating area/ Nice bar area/ Great simple drink menu/ The BEST brick oven pizza with homemade dough!', 1], ['Every time I eat here, I see caring teamwork to a professional degree.', 1], ['Back to good BBQ, lighter fare, reasonable pricing and tell the public they are back to the old ways.', 1], ['Nothing special.', 0], ['The waitress was friendly and happy to accomodate for vegan/veggie options.', 1], ['Great food for the price, which is very high quality and house made.', 1], ['Service is perfect and the family atmosphere is nice to see.', 1], ['The menu had so much good stuff on it i could not decide!', 1], ["I'll definitely be in soon again.", 1], ['Ambience is perfect.', 1], ["I don't each much pasta, but I love the homemade /hand made pastas and thin pizzas here.", 1], ['The yellowtail carpaccio was melt in your mouth fresh.', 1], ['And the beans and rice were mediocre at best.', 0], ['the spaghetti is nothing special whatsoever.', 0], ['The WORST EXPERIENCE EVER.', 0], ['All in all an excellent restaurant highlighted by great service, a unique menu, and a beautiful setting.', 1], ['The sergeant pepper beef sandwich with auju sauce is an excellent sandwich as well.', 1], ["Ordered burger rare came in we'll done.", 0], ["I like Steiners because it's dark and it feels like a bar.", 1], ['This place is awesome if you want something light and healthy during the summer.', 1], ['Now I am getting angry and I want my damn pho.', 0], ["The plantains were the worst I've ever tasted.", 0], ['I think this restaurant suffers from not trying hard enough.', 0], ["Today is the second time I've been to their lunch buffet and it was pretty good.", 1], ['To my disbelief, each dish qualified as the worst version of these foods I have ever tasted.', 0], ['The crÃªpe was delicate and thin and moist.', 1], ["And the red curry had so much bamboo shoots and wasn't very tasty to me.", 0], ["I'd love to go back.", 1], ['seems like a good quick place to grab a bite of some familiar pub food, but do yourself a favor and look elsewhere.', 0], ['They know how to make them here.', 1], ['We were sat right on time and our server from the get go was FANTASTIC!', 1], ['AN HOUR... seriously?', 0], ['Food was delicious!', 1], ['The waitresses are very friendly.', 1], ['The atmosphere is modern and hip, while maintaining a touch of coziness.', 1], ["Sadly, Gordon Ramsey's Steak is a place we shall sharply avoid during our next trip to Vegas.", 0], ["The waiter wasn't helpful or friendly and rarely checked on us.", 0], ['this place is good.', 1], ["The chains, which I'm no fan of, beat this place easily.", 0], ['This place is overpriced, not consistent with their boba, and it really is OVERPRICED!', 0], ['It was a huge awkward 1.5lb piece of cow that was 3/4ths gristle and fat.', 0], ['And it was way to expensive.', 0], ["I gave it 5 stars then, and I'm giving it 5 stars now.", 1], ['Pretty awesome place.', 1], ['The burger is good beef, cooked just right.', 1], ['Food is way overpriced and portions are fucking small.', 0], ['On the ground, right next to our table was a large, smeared, been-stepped-in-and-tracked-everywhere pile of green bird poop.', 0], ['The service was extremely slow.', 0], ['This is a good joint.', 1]]
    1 1.290732163991403e-12
    1 0.013773106062603567
    1 3.2370214587120676e-07
    1 3.1083234426011937e-23
    0 2.4051265169160013e-20
    0 2.798718295649871e-24
    1 4.087458560104883e-21
    0 6.557575894765135e-10
    0 1.8972493550827262e-23
    0 0.011194987775061126
    1 0.00012908988944079476
    0 3.661580427065857e-30
    1 3.839079227114477e-32
    0 4.76480432504439e-07
    1 5.851132312882862e-12
    1 6.710969068331336e-05
    1 0.00014344552906064563
    1 1.4884506210793675e-09
    1 6.659535864328031e-14
    0 2.6208749814231624e-12
    1 2.201792273784673e-22
    1 1.2966080183959605e-09
    1 8.059403224184732e-11
    1 2.9498705305031996e-08
    0 2.2268927244888903e-13
    1 1.5754702228732855e-18
    1 0.0006548950488703734
    1 1.574266944399936e-05
    1 3.148533888799872e-06
    0 5.075515901968883e-16
    1 0.0019269027399455216
    1 1.0626971977784213e-14
    0 1.1019638080627164e-13
    1 2.210456579813941e-18
    1 8.136445210473374e-08
    1 4.4422777197527595e-13
    1 1.858883464943601e-05
    0 1.5816588112118128e-13
    1 1.8498183970400476e-05
    0 4.410786641739989e-35
    0 2.979097307797993e-12
    1 1.949356665009246e-06
    0 0.00014157984817108393
    0 8.80901198769047e-25
    1 6.254923175244836e-11
    1 3.0551269687211095e-08
    1 1.7489997940909286e-20
    1 1.2219682815988523e-10
    1 2.46110302120595e-07
    1 4.3043248100048876e-07
    0 3.972910634009803e-09
    1 3.196526670611082e-30
    0 2.2503455813484323e-14
    1 1.6804459733329157e-19
    1 3.148533888799872e-06
    1 7.272225298747075e-09
    1 4.413426725855503e-12
    1 1.2428613991777851e-11
    1 2.191327739934448e-12
    0 9.877395643757025e-11
    1 0.0004061608716551835
    1 3.9455934931832065e-23
    1 2.702855599816174e-09
    1 3.7555718969489417e-14
    1 5.440642344235434e-07
    0 4.559725750614823e-08
    1 6.131990652697914e-20
    1 1.818242432074653e-17
    0 1.7538207406265747e-15
    0 7.878697282190825e-17
    1 6.024415718087875e-18
    0 1.8983783165424873e-16
    1 1.9067174404643812e-08
    1 5.004998770704692e-17
    1 4.074448493875009e-23
    1 3.2720576760935865e-17
    1 2.5908850288187387e-05
    0 2.0380703214627425e-18
    1 8.814482311346157e-09
    1 4.611118244258313e-29
    0 1.6714230534669552e-13
    1 3.830899708045641e-18
    0 0.25872860635696826
    1 0.0002817738556179126
    1 3.8653865957300574e-05
    1 2.1263835460577988e-08
    1 6.984007788016367e-19
    1 8.96173071992521e-20
    1 7.347210025057458e-05
    0 2.779690823252535e-10
    1 5.274147398564804e-10
    1 3.1526387000181787e-13
    0 0.00012772375602095897
    1 2.890385446147022e-18
    1 7.17387468334148e-06
    1 3.2603753192502035e-09
    0 1.9685701113819558e-10
    0 4.477091848446003e-15
    1 4.4214895392952466e-08
    1 0.0029665818255936367
    accuracy =  0.68
    

# Top 10 words that predicts positive: P[Positive| word] 



```python
import operator
#p[positive|word]p=[word|positive]*p[positive]/p[word]
p_pos_given_word={}
# print(positive_prob)
# print(prob_all_docs)
# print(prob_of_pos)

for word,prob in positive_prob.items():
  p_pos_given_word[word]=positive_prob[word]*prob_of_pos/prob_all_docs[word]
sorted_prob=sorted(p_pos_given_word.items(),key=operator.itemgetter(1),reverse=True)
print("Top 10 words to predict positive: ")
for i in range(0,10):
  print(sorted_prob[i][0])

```

    Top 10 words to predict positive: 
    fantastic.
    spot.
    friendly.
    perfect
    fantastic
    happy
    amazing.
    awesome
    wonderful
    delicious!
    

# Top 10 words that predicts negative : P[negative| word]


```python
import operator
#p[negative|word]p=[word|negative]*p[negative]/p[word]
p_neg_given_word={}

for word,prob in negative_prob.items():
  p_neg_given_word[word]=negative_prob[word]*prob_of_neg/prob_all_docs[word]
sorted_prob=sorted(p_neg_given_word.items(),key=operator.itemgetter(1),reverse=True)
print("Top 10 words to predict negative: ")
for i in range(0,10):
  print(sorted_prob[i][0])
```

    Top 10 words to predict negative: 
    good
    also
    very
    eat
    good.
    service.
    made
    night
    really
    some
    


```python

```



### References:
[1] https://miro.medium.com/max/1200/1*ZW1icngckaSkivS0hXduIQ.jpeg

[2] https://dataaspirant.com/naive-bayes-classifier-machine-learning/#:~:text=Naive%20Bayes%20is%20a%20kind,as%20the%20most%20likely%20class.

[3] https://www.analyticsvidhya.com/wp-content/uploads/2015/09/Bayes_rule-300x172.png

[4] https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-in-training-validation-and-test-set#:~:text=To%20split%20into%20train%20%2F%20validation%20%2F%20test,split%20by%20calling%20scikit-learn%27s%20function%20train_test_split%20%28%29%20twice.





