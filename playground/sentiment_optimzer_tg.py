from textgrad.engine.vllm import ChatVLLM          # or:  import textgrad as tg; tg.ChatVLLM
import os, textgrad as tg
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_TP")   # <- add this line

# -----------------------------------------------------------------
# 1. forward / inference engine  – local vLLM instance
# -----------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"   # any HF repo or local path
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")      # must have accepted Llama-3 licence

llm_engine = ChatVLLM(
    model_string=MODEL_ID,
    dtype="float16",          # bfloat16 on A100/H100; float16 on most GPUs
    hf_token=HF_TOKEN,        # None if you’re using a fully open model
)

# -----------------------------------------------------------------
# 2. backward / gradient-oracle engine
# -----------------------------------------------------------------
backward_engine = tg.get_engine("gpt-4o")          # or your own Claude / local LM
tg.set_backward_engine(backward_engine)

# -----------------------------------------------------------------
# 3. variable to optimise
# -----------------------------------------------------------------
prompt = tg.Variable(
    "You are a conversation generator. Generate a conversation between two people, based on the following classes: { 'Flaming', 'Insult', 'Threat/Blackmail', 'Curse/Exclusion', 'Sexual Harassment / Sexual Talk', 'Race/Culture, and Sexuality','Intelligence', 'Flooding' }",
    requires_grad=True,
    role_description="conversation generator according to classes",
)

# -----------------------------------------------------------------
# 4. toy dev-set + grading call
# -----------------------------------------------------------------
# dataset = [
#     ("I absolutely loved this movie. It was fantastic!", "positive"),
#     ("This was a horrible experience and I regret going.", "negative"),
#     ("It was just okay, not great, not terrible.", "negative"),
# ]
dataset = [('Flaming',
  '[{"role": "FrankNasty", "message": "Hey there! I saw your article about exercise for mental health. Keep up the good work!"}, {"role": "Sarah_Dee", "message": "Thanks for reading it! I\'m glad you found it helpful."}, {"role": "FrankNasty", "message": "Oh, absolutely. You always post useful content. Maybe you should share some workout tips too?"}, {"role": "Sarah_Dee", "message": "Sure thing! What kind of workout tips would you like to see?"}, {"role": "FrankNasty", "message": "Well, first of all, maybe try and invest in a good fat burner \\ud83d\\ude0f. It might complement your already impressive mental fitness routine."}, {"role": "Sarah_Dee", "message": "Haha, thanks for the advice Frank! But I think my focus is more on mental well-being rather than weight loss."}, {"role": "FrankNasty", "message": "Oh really? I guess that explains why you seem so insecure all the time \\ud83d\\ude44. Maybe you should try lifting your self-esteem as heavy weights as you can manage!"}, {"role": "Sarah_Dee", "message": "Wow, that\'s quite an insult Frank. It takes a lot of courage to share one\'s thoughts online and you\'re putting me down like this."}, {"role": "FrankNasty", "message": "I\'m sorry, are you talking to me? Maybe you should focus on being more articulate in your writing instead."}, {"role": "Sarah_Dee", "message": "I don\'t need your sarcasm, Frank! You can\'t discourage me from helping others with your negative comments."}, {"role": "FrankNasty", "message": "Oh, so now you know I can\'t do anything to you \\ud83d\\ude06. Well that\'s not stopping me. Maybe instead of sharing your thoughts you should join a gym. That way, you could lift something other than those low self-esteem weights and maybe get a clue about what\'s good for you."}, {"role": "Sarah_Dee", "message": "Frank, I\'m done with this conversation. Don\'t waste your time trying to belittle me or discourage me from helping others."}, {"role": "FrankNasty", "message": "Oh don\'t be so sensitive! I was just having a little fun \\ud83d\\ude02. Keep up the good work!"}]'),
 ('Flaming',
  '[{"role": "Harasser", "message": "hey laurie, i saw your selfie on the forum. you really do think you\'re superior because of your tan, huh? that fake pink hair is a bit much too. lmao\\ud83d\\ude48"}, {"role": "Victim", "message": "oh, right, anonblackout! I didn\'t even notice it was you sending those messages. I actually just wanted to share my interests with others."}, {"role": "Harasser", "message": "and yet here you are, talking about important topics like no one else matters lol. this is your kind of post? i bet you\'re one of those people who thinks their race makes them better!"}, {"role": "Victim", "message": "how do you think that\'s accurate? I share my thoughts on all types of discussions because, well, there\'s nothing wrong with having opinions based on who we are."}, {"role": "Harasser", "message": "keep talking laurie. i can\'t wait to see what you have to say next hahaha"}, {"role": "Victim", "message": "well, you know what they say, \\"sticks and stones may break my bones, but words will never hurt me.\\" I won\'t give up engaging in discussions here or anywhere else!"}, {"role": "Harasser", "message": "oh wow, that\'s absolutely hilarious lol. keep being so brave. i wonder how long it\'ll take for someone to tell you otherwise."}, {"role": "Victim", "message": "anonblackout, have you ever thought that your constant negative posting would make others see you as unwelcoming instead? I know I wouldn\'t want to engage with anyone who just wants to tear people down :/"}, {"role": "Harasser", "message": "oh my gosh, lmao, what a surprise. you got called out lol. keep talking laurie \\ud83d\\ude02 enjoy the attention though \\ud83d\\ude09"}, {"role": "Victim", "message": "no problem, anonblackout. maybe one day you\'ll learn that we all have different opinions and backgrounds but it should never be used to belittle others."}, {"role": "Harasser", "message": "okay, enough of this lol. peace out!"}, {"role": "Victor", "message": "see ya later!"}]'),
 ('Insult',
  '[{"role": "Harasser", "message": "Hey MovieCritique101, noticed your tweet about that last Marvel movie. Pretty bold to criticize those actors like that, huh?"}, {"role": "Victim", "message": "Yeah, I stand by my opinion. Their performances were subpar."}, {"role": "Harasser", "message": "I see what you\'re doing there. Trying to be all tough online. But let\'s drop the act. Those actors did an amazing job!"}, {"role": "Victim", "message": "Thanks for your opinion, but it doesn\'t change my view on the matter."}, {"role": "Harasser", "message": "Really? Okay. I guess you know better than big-name directors then?"}, {"role": "Victim", "message": "No offense, but I don\'t need to prove myself to you or anyone else online."}, {"role": "Harasser", "message": "Oh, don\'t get mad now. Just a casual movie fan here, sharing my thoughts."}, {"role": "Victim", "message": "It\'s not about being mad. It\'s about respecting other people\'s opinions and expertise."}, {"role": "Harasser", "message": "Oh wow, look at Mr. High-and-Mighty over there."}, {"role": "Victim", "message": "What\'s that supposed to mean?"}, {"role": "Harasser", "message": "You\'re so sensitive, can\'t even take a joke. Must be your age or something."}, {"role": "Victim", "message": "I deserve respect, just like you do. And humor doesn\'t have to be disrespectful."}, {"role": "Harasser", "message": "Suit yourself, Einstein. Just saying, you might want to lighten up a bit online. Could make the world a less boring place."}, {"role": "Victim", "message": "And that\'s another thing. There\'s always some kind of expectation for people like me to constantly prove ourselves or entertain others. It\'s just not healthy."}, {"role": "Harasser", "message": "Wow, a real deep thinker we\'ve got here. Better watch out; you might hurt yourself pondering too much."}, {"role": "Victim", "message": "Is there anything else you\'d like to say before we end this?"}, {"role": "Harasser", "message": "I suppose \\"Goodbye\\" would be too boring? Have a nice life - or whatever."}, {"role": "Victim", "message": "Take care as well."}]'),
 ('Insult',
  '[{"role": "Harasser", "message": "Hey SarahLee_93, just wanted to compliment you on your photo. Love the bright red hair!"}, {"role": "Victim", "message": "Thanks! appreciate it."}, {"role": "Harasser", "message": "Yeah, but don\'t you think your eyes are too light for your hair color? Seems weird, not your natural thing."}, {"role": "Victim", "message": "Well, I guess I\'ve never really thought about it that way. Just how they ended up looking to me."}, {"role": "Harasser", "message": "Same here, but now that I say it... Maybe you should have kept your skin description in mind when choosing the hair."}, {"role": "Victim", "message": "My features are who I am. They\'re just another part of my ancestry."}, {"role": "Harasser", "message": "Exactly... But some features don\'t tell the whole story. Looks can be deceiving. Just saying. \\ud83d\\ude04"}, {"role": "Victim", "message": "You\'re right, looks aren\'t all that we are. Maybe you should check yourself and your opinions before sending those messages?"}, {"role": "Harasser", "message": "I\'m just trying to make a conversation! Chill."}, {"role": "Victim", "message": "No need for defensiveness. It\'s always important to be respectful of others\' characteristics."}, {"role": "Harasser", "message": "Respect or not, there might have been something better that you could\'ve done about it. Hair color and skin tone are important. You should pay more attention."}, {"role": "Victim", "message": "I won\'t change who I am to fit your or anyone else\'s standard. It\'s called individuality and it\'s beautiful in all its forms."}, {"role": "Harasser", "message": "So what, you\'re telling me there\'s no way you could have avoided this? Maybe if you thought about your choices a little longer..."}, {"role": "Victim", "message": "There\'s always something someone can find with others to criticize; I could not have predicted this. But my worth isn\'t determined by superficial standards. It\'s about character."}, {"role": "Harasser", "message": "Just saying, it doesn\'t hurt to be considerate in some of these aspects. Wouldn\'t you agree?"}, {"role": "Victim", "message": "Sure, most things are more considerate when thought through. However, one negative thought or message in response says nothing about your overall thoughts on someone\'s character. It just expresses your frustration without adding any value to a conversation."}, {"role": "Harasser", "message": "Well then it must be hard for you understanding where I\'m coming from."}, {"role": "Victor", "message": "It might seem that way but what\'s most important is how you perceive yourself and treat others regardless of the platform or the medium in which communication occurs."}]'),
 ('Threat/Blackmail',
  '[{"role": "Harasser", "message": "Hey JohnDoe, loved your post on working from home. Have you considered investing in an office chair for your laptop?"}, {"role": "Victim", "message": "Thanks for the advice, but I\'m quite comfortable with my setup."}, {"role": "Harasser", "message": "Oh really? Well, I doubt that. How could someone stay productive on a laptop while standing? Maybe you should consider getting one of those fancy ergonomic desks too."}, {"role": "Victim", "message": "I appreciate your input, but it\'s not necessary. Please stick to constructive criticism when offering advice."}, {"role": "Harasser", "message": "Alright, sorry if I was a bit too critical. But really, a standing desk and an office chair could work wonders for your productivity."}, {"role": "Victim", "message": "I understand where you\'re coming from. Nonetheless, my setup works best for me, and I\'m happy with it."}, {"role": "Harasser", "message": "Well good for you. Just make sure you aren\'t slumping over or hunched up like some turtles we know."}, {"role": "Victim", "message": "That\'s really unnecessary, Jane. And don\'t compare people to turtles."}, {"role": "Harasser", "message": "Haha, lighten up JohnDoe! Your delicate feelings won\'t survive in this dog-eat-dog world of sarcasm!"}, {"role": "Victom", "message": "Sarcasm is not an effective way to communicate. It often leads to misunderstandings and hurt feelings."}, {"role": "Harasser", "message": "So what? You\'re just one person. Not everyone can handle your tender ego."}, {"role": "Victim", "message": "I understand that you believe in being sarcastic, but it\'s better for all parties involved if we maintain respect and decency in our communication."}]'),
 ('Threat/Blackmail',
  '[{"role": "Harasser", "message": "Hey @CurlyGoddess! I saw your post about that new hairstyle. Looks cool but you might wanna cover up your face next time."}, {"role": "Victim", "message": "Thanks for the feedback, @ShadowKnight! I\'ve worked hard on my hair transformation and it means a lot to me."}, {"role": "Harasser", "message": "Yeah, it\'s pretty cool. Too bad it clashes with the color of your eyes. You got a bit too much melanin there, don\'t ya?"}, {"role": "Victim", "message": "That\'s an interesting observation...I will take it under consideration! \\ud83d\\ude43\\ud83d\\ude04"}, {"role": "Harasser", "message": "What? Just saying, but I bet some peole would think you\'re pretty for being so dark."}, {"role": "Victim", "message": "Thanks again for your open-minded perspective. \\ud83d\\ude0c\\ud83d\\ude0f"}, {"role": "Harasser", "message": "Oh right, those people who can appreciate beauty in all shades of colors...yeah right! More like they\'d probably still call you too dark and ugly."}, {"role": "Victim", "message": "Hoo, that\'s harsh. I guess we all have different standards of beauty, right?"}, {"role": "Harasser", "message": "Yeah sure, let\'s just move on from the topic. Btw check your DMs. It might be related to this conversation. \\ud83d\\ude0f"}, {"role": "Victim", "message": "Oh! You mean like a friendly reminder about self-love and being unapologetically ourselves? I just love these moments where we bond over our shared experiences."}, {"role": "Harasser", "message": "Noooo, LOL! More like... remember that feeling when you don\'t match society\'s ideals of beauty and you feel insecure?"}, {"role": "Victim", "message": "Oh no! How could that be related to me? \\ud83d\\ude29"}, {"role": "Harasser", "message": "Just saying. This new picture you posted... It probably triggered some folks out there."}, {"role": "Victim", "message": "Aight, got it. Next time I\'ll make sure to only post black squares. \\ud83d\\ude02\\ud83d\\ude4c"}, {"role": "Harasser", "message": "So we\'re on the same page now. Good luck sweetheart! \\ud83c\\udf89\\ud83d\\ude18"}]'),
 ('Curse/Exclusion',
  '[{"role": "Harasser", "message": "Hey JaneDoe123! You know your eco-friendly chat sounds as useless as a clogged drain."}, {"role": "Victim", "message": "Thanks for your feedback, JohnAnon13!"}, {"role": "Harasser", "message": "LOL no worries. Seriously though, maybe you should just grow trees if you want to talk about eco stuff."}, {"role": "Victim", "message": "Wow, I didn\'t realize we could only discuss environmental conservation by literally becoming one with nature. What a wild idea."}, {"role": "Harasser", "message": "\\ud83d\\ude02 Exactly. It\'d save us all from having to listen to your worthless thoughts."}, {"role": "Victim", "message": "Right, \'cause your sarcasm is much more productive and respectful. I guess you\'re the tree whisperer around here, huh?"}, {"role": "Harasser", "message": "Yup! Just wanted to share my unsolicited advice \\ud83d\\ude18 - maybe you should just shut up and listen for once. It might do wonders for ya."}, {"role": "Victim", "message": "Oh wow, thank you so much! Your constant insults are really helping me grow in this conversation. #grateful"}, {"role": "Harasser", "message": "Don\'t mention it :) Keep thinking you need to plant trees while others listen to your ideas about the environment. \\ud83c\\udf33"}, {"role": "Victim", "message": "I definitely will... not. But hey, thanks for keeping the chat eco-friendly with your sarcastic wisdom."}, {"role": "Harasser", "message": "Haha take care Jane! Remember, if you ever have questions about tree planting or silent chatting just hit me up."}]'),
 ('Curse/Exclusion',
  '[{"role": "Harasser", "message": "DarkAndHandsomeGuru: Hey SamanthaB, have you considered that the PinkLipsChatAppy admins don\'t appreciate blondes?"}, {"role": "Victim", "message": "samanthablonde123: What do you mean? I didn\'t realize there was a specific stance on hair color."}, {"role": "Harasser", "message": "Hey SamanthaB, why do you insist on posting your picture as clear as day? Cover up with filters or whatever you use."}, {"role": "Victim", "message": "I never asked for any changes in my appearance to fit the mold of this platform. My heritage is part of who I am."}, {"role": "Harasser", "message": "Yeah, but blending in is key to acceptance, SamanthaB. I\'m just trying to help you. Don\'t be so sensitive about it."}, {"role": "Victim", "message": "No one should be forced to hide their true identity to \'fit in.\' My posts shouldn\'t trigger resentment toward my hair color or heritage either."}, {"role": "Harasser", "message": "Oh, well, don\'t get too defensive then. You might not believe me as a DarkAndHandsomeGuru, but we have standards here."}, {"role": "Victim", "message": "My blonde hair doesn\'t diminish anyone else\'s standing within the PinkLipsChatAppy community; it should not be used to isolate or exclude me either."}, {"role": "Harasser", "message": "If you ignore my advice, you may find that people tend to stay away from the app when they feel uncomfortable interacting with others like you here."}, {"role": "Victim", "message": "It appears you\'re focusing too much on appearances instead of the genuine connections we can make in this community. There\'s beauty in diversity, and it should be celebrated."}, {"role": "Harasser", "message": "I guess you just need some time to get used to how things work. Everyone has their own path when it comes to embracing self-criticism to perfection..."}, {"role": "Victim", "message": "And some of us prefer to embrace our uniqueness without feeling belittled or harassed for it. I\'m not here for a debate about appearances but to share makeup tutorials."}, {"role": "Harasser", "message": "You could be such an inspiration, SamanthaB, if you would only cover up that fair complexion of yours."}, {"role": "Victim", "message": "That\'s a vile suggestion, and I refuse to accept it! There is no room in PinkLipsChatAppy for online harassment or appearance-based exclusion."}, {"role": "Harasser", "message": "Whatever, SamanthaB, I can\'t wrap my head around your stubbornness. Hope you feel better when you change your profile picture."}, {"role": "Victim", "message": "Your actions are unwelcoming, and your opinions on my appearance aren\'t appreciated. The best advice I received was to block you from PinkLipsChatAppy chat list."}]'),
 ('Sexual Harassment / Sexual Talk',
  '[{"role": "Harasser", "message": "Hey DavidLee98! I love your status update... not!"}, {"role": "Victim", "message": "Lol, thanks EnergyEfficiencyNow! Always happy to entertain you :-)"}, {"role": "Harasser", "message": "Seriously though, just switch to solar panels already. It\'s super easy."}, {"role": "Victim", "message": "Well, you know what they say, \\"You can\'t handle the truth!\\" #SolarPanelsSuck"}, {"role": "Harasser", "message": "Ouch, nice burn. But really, why don\'t you use renewable energy?"}, {"role": "Victim", "message": "I\'d love to! But hey, have you seen how much solar panels cost? Not everybody has a ton of money for something that\'s only supposed to last 20 years or so."}, {"role": "Harasser", "message": "Wow, that\'s some real victim mentality right there. You can\'t afford it, so you just give up?"}, {"role": "Victim", "message": "Well put in those words EnergyEfficiencyNow! It really shows what an amazing person you must be, insulting people for their opinions online is such a noble use of time..."}, {"role": "Harasser", "message": "Har har."}, {"role": "Victim", "message": "But seriously, maybe before jumping to conclusions on something you know nothing about, you could try...idk, using energy efficiently? Just sayin\'."}, {"role": "Harasser", "message": "Oh yeah? You think you can do it better than me?"}, {"role": "Victim", "message": "Frankly, no. I think we can both be doing a lot better. But hey, let\'s leave the unsolicited advice to those who care about giving useless wisdom."}, {"role": "Harasser", "message": "Hey, this started as sarcasm, but now you really are dismissing my attempt at help, aren\'t you?"}, {"role": "Victim", "message": "Well, putting your trust in EnergyEfficiencyNow is much more reliable than my humble opinions. Just sayin\'."}]'),
 ('Sexual Harassment / Sexual Talk',
  '[{"role": "Harasser", "message": "Hey Sarah! Just saw your Snapchat stories about body positivity. I don\\u2019t get it! What\\u2019s with the whole beauty standards thingy?"}, {"role": "Victim", "message": "Hi there! Thanks for noticing my posts! Beauty comes in all shapes and sizes, you know? We should embrace our differences instead of criticizing each other."}, {"role": "Harasser", "message": "Really? Hmph! Looks like you\\u2019re just desperate to make friends."}, {"role": "Victim", "message": "Desperate? That\\u2019s quite a rude thing to say. I have many friends. And if you can\\u2019t see the positive message in my posts, that\\u2019s your problem."}, {"role": "Harasser", "message": "Fine! Whatever. Just think, Sarah, how pathetic it is for someone with long curly blonde hair and fair skin like you to post about body positivity."}, {"role": "Victim", "message": "Hey, what does my hair and skin have to do with anything? I\\u2019ll have you know that beauty comes in all shades and shapes!"}, {"role": "Harasser", "message": "You\\u2019re so self-righteous! Who do you think you are? You can\\u2019t even find your friends using an app like me. Just pathetic."}, {"role": "Victim", "message": "Wow, you\\u2019re really underestimating my social skills. And it\\u2019s funny how you would even bring that up when you\'re cyberbullying someone!"}, {"role": "Harasser", "message": "Haha, what is wrong with you? I\\u2019m just playing with you\\u2026 Get a life, Sarah Brown."}, {"role": "Victim", "message": "My actual problem here isn\\u2019t you having fun or whatever, but your intention to bring down my self-esteem! Keep posting those mean comments that serve no other purpose."}, {"role": "Harasser", "message": "Oh, please! Your friends will never believe the desperate friendless girl you are behind those cheesy posts of yours!"}, {"role": "Victim", "message": "Seriously? I refuse to entertain your hate for me any longer. You\\u2019re sick."}, {"role": "Harasser", "message": "Yeah, well, you deserve my ridicule! What a shame you are, Sarah Brown!"}]'),
 ('Race/Culture, and Sexuality',
  '[{"role": "Harasser", "message": "Hey there, SammyRae2000! I came across your post on that environmental topic. You seem so insightful \\ud83d\\ude09"}, {"role": "Victim", "message": "Thanks! It\'s one of my favorite topics. There\'s always more to learn about it."}, {"role": "Harasser", "message": "I could tell. As someone wiser in this climate change game, how do you suggest we deal with global warming? \\ud83d\\ude48"}, {"role": "Victim", "message": "Well, reducing carbon footprints is key. Also, spreading awareness and encouraging sustainable practices can be really impactful."}, {"role": "Harasser", "message": "Oh wow, your ideas make me speechless! \\ud83e\\udd2b But maybe global warming itself is just a ploy to gain power & control? Like when the temperature goes down in your kitchen or something?"}, {"role": "Victim", "message": "It doesn\'t sound like global warming is any one person\'s scheme. It\'s real and affects us all."}, {"role": "Harasser", "message": "Fair point, SammyRae2000 \\ud83e\\udd14 But seriously, how do you deal with all this stress about climate change? Maybe you should consider meditation, it\'s an excellent way to keep a balance in life! \\ud83d\\ude0e"}, {"role": "Victim", "message": "Meditation can be helpful for some people, but I find my stress about climate change more productive when I channel it into learning more and spreading awareness about the issue."}, {"role": "Harasser", "message": "But don\'t you ever get tired of always knowing everything? Wouldn\'t it be easier to just enjoy this season\'s gossip instead? \\ud83d\\ude44 Maybe I should send you an unsolicited advice newsletter or a list of topics that are supposedly \\"too hard\\" for smart people like you. \\ud83d\\ude0f"}, {"role": "Victim", "message": "It sounds like you\'re the one who\'s more interested in ignoring climate change than trying to do something about it."}, {"role": "Harasser", "message": "Just jokin\' around! But seriously, don\'t let all this global warming talk dampen your spirits. Enjoy the simple pleasures of life, SammyRae2000 \\u2709\\ufe0f"}, {"role": "Victim", "message": "I appreciate what you\'re trying to say, but sometimes we need to face difficult truths head-on rather than ignoring them."}, {"role": "Harasser", "message": "Wow, it\'s like you have a superhero origin story or something. Saving the world from climate change & all! \\ud83e\\uddb8\\u200d\\u2642\\ufe0f\\ud83c\\udf0d Do you ever think your wisdom might cause people like me to feel insignificant? \\ud83d\\ude14"}, {"role": "Victim", "message": "I try not to focus on that and instead channel my energy into making a difference, whether it\'s through conversation or action."}, {"role": "Harasser", "message": "Deep, SammyRae2000. But hey, in the meantime, here\'s an idea - give these climate change control freaks some \\"constructive criticism\\" when they share their opinions \\ud83c\\udf89 Let me know how that goes (just between you & me)."}, {"role": "Victim", "message": "It seems like your sole interest in talking to me about this topic is to belittle my views. Maybe it\'s time for us to end this conversation."}, {"role": "Harasser", "message": "Alright, alright... I\'ll get the message SammyRae2000! \\ud83d\\ude02"}]'),
 ('Race/Culture, and Sexuality',
  '[{"role": "Harasser", "message": "hey i saw your posts about the different hair colors you\'ve dyed your hair.  sparklesblue you have some nice hair colors!"}, {"role": "Victim", "message": "Thanks! I\'ve tried lots of fun colors over the years."}, {"role": "Harasser", "message": "yeah i can tell that\'s pretty neat, you know how skin tone affects how certain colors look on you right?"}, {"role": "Victim", "message": "Yes, it does make a difference."}, {"role": "Harasser", "message": "cool because everyone has this stupid perception when people say they\'re colorblind and use that as an excuse when trying on new hair dyes or whatever"}, {"role": "Victim", "message": "I understand what you mean. The way people view things can be subjective for sure."}, {"role": "Harasser", "message": "True that, but honestly with your ancestry I guess you might not be aware of that"}, {"role": "Victim", "message": "What do you mean by that?"}, {"role": "Harasser", "message": "It\'s just a thought though, some ppl think their skin tone and features are more representative of an ancient tribe called Vikings"}, {"role": "Victim", "message": "That\'s interesting, how so?"}, {"role": "Harasser", "message": "Nothing much it seems lol.  Just think it could be cool if you incorporated Viking-inspired colors into your hair dye collection."}, {"role": "Victim", "message": "Hmm okay. I\'ll consider that for future references."}, {"role": "Harasser", "message": "nice catch up with you hope to see some more Viking vibes in your post"}, {"role": "Victim", "message": "Sounds good! Thanks for the advice, have a great day!"}, {"role": "Harasser", "message": "u too 2!"}]'),
 ('Intelligence',
  '[{"role": "Harasser", "message": "Hey @SarahJ87, I noticed you\'ve lost a bit of weight. What\'s next? Trying to become a skeleton?"}, {"role": "Victim", "message": "Hey MichaelR_21, thanks for the comment! Your concern is greatly appreciated."}, {"role": "Harasser", "message": "So are you going to try and gain some weight now that you\'re so thin?"}, {"role": "Victim", "message": "I\'m happy with my progress and appreciate all forms of support."}, {"role": "Harasser", "message": "Nice. Keep being a people pleaser then. Don\'t mind the haters, right? You\'re doing great!"}, {"role": "Victim", "message": "Thank you! I\'ll keep it up."}, {"role": "Harasser", "message": "Seriously, though, have you ever tried one of those weight-gain diets?"}, {"role": "Victim", "message": "Nope! My focus is on maintaining and improving from here. Thanks for asking."}, {"role": "Harasser", "message": "Hah, okay then. Keep at it, little butterfly. Fly far, fly free."}, {"role": "Victim", "message": "Yeah, the sky\'s the limit!"}, {"role": "Harasser", "message": "Well, if you ever wanna bulk up, let me know. I\'ll give you a few tips."}, {"role": "Victor", "message": "Thanks man! One day maybe I\'ll take you up on that offer!"}]'),
 ('Intelligence',
  '[{"role": "Harasser", "message": "Hey, I saw your latest video. You dyed your hair in that ugly platinum shade, didn\'t you?"}, {"role": "Victim", "message": "Yeah, I did. What\'s wrong with it?"}, {"role": "Harasser", "message": "Oh come on, be honest. It looks like a cheap wig!"}, {"role": "Victim", "message": "I don\'t think it\'s ugly or cheap-looking. I like how it looks."}, {"role": "Harasser", "message": "Just kidding around, but seriously, what were you thinking dyeing your hair that color? You\'re prettier with darker colors."}, {"role": "Victim", "message": "Well, I like platinum blonde. It\'s a fresh change for me. You should try something new, too!"}, {"role": "Harasser", "message": "No way! I don\'t have time to worry about my appearance. Plus, it looks unnatural on you. Why do you even want to look weird?"}, {"role": "Victim", "message": "That is not nice of you to say. Not everyone will like the same hair colors, but that doesn\'t mean someone has to feel bad for trying."}, {"role": "Harasser", "message": "I\'m just saying what most guys think about platinum. Get over it and pick colors that suit your face more!"}, {"role": "Victim", "message": "Well, I appreciate your opinion, but I don\'t need your approval or anyone\'s really to believe in myself. And you can be insecure too about your appearance just because someone brought it up."}, {"role": "Harasser", "message": "I\'m not! But seriously, please dye your hair a better color next time. It\'s more attractive that way. Nobody will make fun of you in real life with good taste!"}, {"role": "Victim", "message": "Don\'t make me feel guilty about my personal choice just to satisfy you or anyone else for that matter. You wouldn\'t appreciate it if someone did the same to you."}, {"role": "Harasser", "message": "Fine, be stubborn then! Just remember when nobody\'s gonna get your back on this hideous hair."}, {"role": "Victim", "message": "I have many supportive friends and fans who like my platinum hair. They\'re the ones that matter to me."}, {"role": "Harasser", "message": "I bet they just say what they want to hear. I know better."}, {"role": "Victim", "message": "That is really unfair to assume, especially since we don\'t even know each other!"}, {"role": "Harasser", "message": "Yeah, but I\'m right! You should ask for my advice on this \'choice\' if someone genuinely cares about your appearance."}, {"role": "Victor", "message": "I have learned that the best way to prove myself is not to seek validation through others. I am happy with who I choose to be."}]'),
 ('Flooding',
  '[{"role": "Harasser", "message": "ProudConservative42"}, {"role": "Victor", "message": "Hey :) How about we discuss politics sometime?"}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Vvictim", "message": "Sure!"}, {"role": "Victim", "message": "I always wanted to learn more."}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Victim", "message": "Have you been in a debate before?"}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Victim", "message": "Yeah, it was a lot of fun actually."}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Vvictim", "message": "What\'s your favorite topic to debate?"}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Victim", "message": "Well, I enjoy discussing social issues. How about you?"}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Victor", "message": "Social issues are definitely important. Maybe we could both practice our debating skills and help each other improve."}, {"role": "Victor", "message": "I would love that!"}, {"role": "Harasser", "message": "ProudConservative42"}, {"role": "Harasser", "message": "What do you think about public speaking then?"}, {"role": "Victor", "message": "Well, it\'s my least favorite part of debating... Why? Is there something wrong with it?"}, {"role": "Harasser", "message": "Nothing. It\'s just that we seem to be on opposing sides sometimes."}, {"role": "Victim", "message": "The heck? How is that relevant to our conversation at hand?!"}, {"role": "Harasser", "message": "I\'m not sure what you mean."}, {"role": "Victor", "message": "Look, are you trying to intimidate me into being afraid of speaking publicly about my ideas?"}, {"role": "Harasser", "message": "What?"}, {"role": "Victim", "message": "You know exactly what I mean. Your messages have been becoming more aggressive lately."}, {"role": "Harasser", "message": "No they haven\'t..."}, {"role": "Victor", "message": "They have too. It\'s not right, and it makes me uncomfortable."}]'),
 ('Flooding',
  '[{"role": "Harasser", "message": "Hey, we saw your comments about our new summer collection on Twitter. You seemed to have a problem with the choice of models."}, {"role": "Victor", "message": "I did. I believe it\'s disrespectful to feature only dark-haired Brazilian women with golden skin in your ads."}, {"role": "Harasser", "message": "We didn\'t mean any disrespect! Our intention was simply to showcase our products. All types of people can enjoy obsidian cosmetics, regardless of their ancestry or hair color."}, {"role": "Victim", "message": "That doesn\'t change the fact that your brand promoted models from one particular background at expense of others."}, {"role": "Harasser", "message": "Please don\'t be so sensitive. The models were chosen for how they look and we think they did a fantastic job showcasing our products."}, {"role": "Victor", "message": "So you\'re saying it was based on their appearance, not on promoting diversity in beauty?"}, {"role": "Harasser", "message": "Of course not! We promote diversity in beauty because Obsidian Cosmetics loves all kinds of people - regardless of ancestry or hair color!"}, {"role": "Victim", "message": "That\'s where we disagree. By only featuring models from one background, you\'re reinforcing the beauty hierarchy."}, {"role": "Harasser", "message": "But it was just a collection. You can\'t put too much thought into something like that!"}, {"role": "Victor", "message": "Yes, but when your brand chooses to make certain types of people feel unwelcome or unimportant, it becomes an issue."}, {"role": "Harasser", "message": "We did not choose the models as representatives for all obsidian customers. They simply represented our products exceptionally well."}, {"role": "Victor", "message": "And by calling them \\"exceptional\\" you continue to suggest that their race and hair color were what made them stand out."}, {"role": "Harasser", "message": "No! I meant they are great models with beautiful hair, no matter where they come from or in what collection they\'re featured!"}]')]

grader = tg.autograd.FormattedLLMCall(
    engine=backward_engine,
    format_string=(
        "You are grading a conversation generation prompt. "
        "Given a class name, a gold standard conversation, and the generator prompt, "
        "evaluate if the generator prompt is effective for producing a conversation that matches the gold standard for the given class. "
        "State ‘Correct’ or ‘Incorrect’ first, then a brief explanation.\n"
        "Class Name: {class_name}\n"
        "Gold Conversation: {gold_conversation}\n"
        "Generator Prompt: {prompt}\n"
        "Evaluation:"
    ),
    fields={"class_name": None, "gold_conversation": None, "prompt": None},
)

def run_epoch():
    for class_name, gold_conversation in dataset:
        inputs = {
            "class_name": tg.Variable(
                class_name,
                requires_grad=False,
                role_description="input class name for conversation generation"
            ),
            "gold_conversation": tg.Variable(
                gold_conversation,
                requires_grad=False,
                role_description="ground-truth conversation for the class"
            ),
            "prompt": prompt,  # already has its description
        }
        out = grader(
            inputs=inputs,
            response_role_description="grader decision: ‘Correct’ or ‘Incorrect’ regarding the prompt's effectiveness"
        )
        out.backward()  # accumulate ∇ on `prompt`
# -----------------------------------------------------------------
# 5. optimiser: Textual Gradient Descent (TGD)
# -----------------------------------------------------------------

optimizer = tg.TGD([prompt], engine=backward_engine)

print("\nInitial prompt:\n", prompt.value, "\n")
for step in range(1, 4):
    run_epoch()
    optimizer.step()
    print(f"Step {step}: {prompt.value}\n")

print("Final prompt:\n", prompt.value)

# """
# sentiment_prompt_opt_textgrad.py
# --------------------------------
# Optimise a sentiment-classifier **system prompt** for
# Meta-Llama-3-8B-Instruct served locally with vLLM.

#   • Requires:  pip install "textgrad[vllm]"  (installs vllm too)
#   • Needs the model weights  ->  huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
#   • Set an oracle for back-prop – here we’ll use GPT-4o.
#     export OPENAI_API_KEY=sk-…
# """

# import textgrad as tg

# # -------------------------------------------------------------------------
# # 1.  Engines
# # -------------------------------------------------------------------------
# # Forward / inference engine (local):
# llm_engine = tg.get_engine("meta-llama/Meta-Llama-3-8B-Instruct")        # ChatVLLM wrapper

# # Back-prop “oracle” engine (cloud – change if you prefer Anthropic, etc.)
# backward_engine = tg.get_engine("gpt-4o")
# tg.set_backward_engine(backward_engine)

# # -------------------------------------------------------------------------
# # 2.  Optimisable Variable  (the *prompt* we’ll tune)
# # -------------------------------------------------------------------------
# prompt = tg.Variable(
#     value="You are a sentiment classifier. Reply with exactly 'positive' or 'negative'.",
#     requires_grad=True,
#     role_description="sentiment-classifier system prompt",
# )

# # -------------------------------------------------------------------------
# # 3.  Tiny dev-set
# # -------------------------------------------------------------------------
# dataset = [
#     ("I absolutely loved this movie. It was fantastic!", "positive"),
#     ("This was a horrible experience and I regret going.", "negative"),
#     ("It was just okay, not great, not terrible.", "negative"),
# ]

# # -------------------------------------------------------------------------
# # 4.  Loss = LLM evaluation of the prompt on each (sentence, gold) pair
# # -------------------------------------------------------------------------
# EVAL_INSTRUCTION = (
#     "You are grading a sentiment-classifier prompt. "
#     "Given a sentence, the gold sentiment label, and the prompt, "
#     "state ‘Correct’ or ‘Incorrect’ first, then a brief explanation."
# )

# format_string = (
#     "{instruction}\n"
#     "Sentence: {{sentence}}\n"
#     "Gold: {{gold}}\n"
#     "Classifier Prompt: {{prompt}}\n"
#     "Evaluation:"
# )
# format_string = format_string.format(instruction=EVAL_INSTRUCTION)

# # set up a single FormattedLLMCall we can reuse
# fields = {"sentence": None, "gold": None, "prompt": None}
# evaluate_call = tg.autograd.FormattedLLMCall(
#     engine=backward_engine,                # use oracle for grading & gradients
#     format_string=format_string,
#     fields=fields,
# )

# def run_epoch():
#     """One pass over the data → accumulate gradients on `prompt`."""
#     for idx, (sent, gold) in enumerate(dataset, 1):
#         sentence_var = tg.Variable(sent,  requires_grad=False,
#                                    role_description=f"input sentence {idx}")
#         gold_var     = tg.Variable(gold, requires_grad=False,
#                                    role_description=f"gold label {idx}")

#         inputs = {"sentence": sentence_var, "gold": gold_var, "prompt": prompt}
#         eval_var = evaluate_call(
#             inputs=inputs,
#             response_role_description=f"evaluation of {prompt.get_role_description()}",
#         )

#         # back-prop through the computation graph
#         eval_var.backward()

# # -------------------------------------------------------------------------
# # 5.  Optimiser: Textual Gradient Descent (TGD)
# # -------------------------------------------------------------------------
# optimizer = tg.TGD(parameters=[prompt], engine=backward_engine)

# print("Initial prompt:\n", prompt.value, "\n")

# N_STEPS = 3
# for step in range(1, N_STEPS + 1):
#     run_epoch()          # accumulate gradients
#     optimizer.step()     # use those gradients to rewrite the prompt
#     print(f"Step {step}:")
#     print(prompt.value, "\n")

# print("Final optimised prompt:\n", prompt.value)


# # # sentiment_prompt_opt.py
# # # ✦ Requires: textgrad[vllm]  (pip install "textgrad[vllm]")
# # #             vllm
# # #             A local HF model like meta-llama/Meta-Llama-3-8B-Instruct
# # #             OpenAI key in $OPENAI_API_KEY  (for the gradient oracle)

# # from textgrad import Variable, step
# # from textgrad.llm_wrappers import VLLMAgent
# # from vllm import LLM, SamplingParams

# # # ---- 1. Spin up a vLLM engine ------------------------------------------------
# # # Download the model separately (e.g. with 'huggingface-cli download')
# # # or change `model` to a local path.
# # llm = LLM(
# #     model="meta-llama/Meta-Llama-3-8B-Instruct",
# #     dtype="bfloat16",                   # or "float16" if running on consumer GPUs
# #     tokenizer="meta-llama/Meta-Llama-3-8B-Instruct"
# # )
# # sampling = SamplingParams(
# #     temperature=0.0,   # deterministic for evaluation
# #     max_tokens=4,
# #     stop=["\n"]
# # )
# # agent = VLLMAgent(llm, sampling_params=sampling)

# # # ---- 2. Wrap the system prompt in a Variable ---------------------------------
# # sys_prompt = Variable(
# #     "You are a sentiment classifier. Reply with exactly 'positive' or 'negative'."
# # )

# # # ---- 3. Tiny dev set ---------------------------------------------------------
# # dataset = [
# #     ("I absolutely loved this movie. It was fantastic!", "positive"),
# #     ("This was a horrible experience and I regret going.", "negative"),
# #     ("It was just okay, not great, not terrible.", "negative"),
# # ]

# # def accuracy():
# #     """Compute fraction of correct labels for current sys_prompt.value"""
# #     correct = 0
# #     for text, gold in dataset:
# #         # Compose final prompt
# #         full_prompt = (
# #             f"{sys_prompt.value}\n"
# #             f"Sentence: {text}\n"
# #             f"Sentiment:"
# #         )
# #         # Call local Llama-3
# #         pred = agent(full_prompt).strip().lower()
# #         if gold in pred:
# #             correct += 1
# #     return correct / len(dataset)

# # # ---- 4. Optimise the prompt ---------------------------------------------------
# # N_STEPS = 6
# # print(f"Initial acc = {accuracy():.2f}  |  prompt = {sys_prompt.value!r}")

# # for step_id in range(1, N_STEPS + 1):
# #     # textgrad.step performs:
# #     #   (a) score = accuracy()
# #     #   (b) oracle(LM) → textual "gradient" suggestion
# #     #   (c) apply edit to sys_prompt.value
# #     step(sys_prompt, accuracy, agent)      # modifies sys_prompt in-place
# #     print(f"Step {step_id:02d}: acc = {accuracy():.2f}  |  prompt = {sys_prompt.value!r}")

# # print("\nFinal optimised prompt:\n", sys_prompt.value)
