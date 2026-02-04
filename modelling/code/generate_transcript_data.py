"""
Generate ENT call transcript training data for LLM finetuning.
Transcripts mimic real patient/caller phone conversations; output is urgency only.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "training_data.jsonl"

INSTRUCTION = "You are an ENT triage expert. Classify the urgency of this patient as routine, semi-urgent, or urgent based on their symptoms."


def transcript(parts):
    """Join conversation parts into one transcript block (what the LLM will see)."""
    return " ".join(parts).replace("  ", " ").strip()


# --- ROUTINE (mild, stable, improving, no red flags) ---
ROUTINE_TRANSCRIPTS = [
    "Caller: Hi, I've had a bit of a sore throat for like two days. It's not too bad, more like a scratch. No fever or anything. I've been drinking tea and it seems to be getting a little better.",
    "Patient: Yeah so I have some congestion and a runny nose. Started maybe four days ago. It's kind of annoying but I'm not really sick. No ear pain, no fever. Just wondering if I need to come in or if it'll clear on its own.",
    "Caller: My son has had a mild earache for a day. He's not crying or anything, playing normally. No fever. I just wanted to check if we should bring him in or wait.",
    "Patient: I get sinus pressure every year around this time. It's the same as always, just congestion and pressure in my face. No fever, no severe pain. Wanted to see if I need an appointment or just use the usual stuff.",
    "Caller: Hi, I've had a hoarse voice for about three days. I had a cold last week and it's kind of left me with this. No trouble swallowing or breathing. It's actually improving a bit.",
    "Patient: So I have this thing where my ear feels a bit full, like there's pressure. No pain really, no hearing loss. It's been maybe five days. No fever.",
    "Caller: My daughter has a mild sore throat. She's eating and drinking fine, no fever. It's been two days. She said it's getting better today. Do we need to see someone?",
    "Patient: I've had a bit of a cough and post-nasal drip for a week. No fever, no shortness of breath. Just annoying. Wondering if I need to be seen.",
    "Caller: I have allergies I think. Runny nose, sneezing, itchy throat. It's the same every spring. No fever, no ear pain. Just checking if I need an ENT visit or if OTC is fine.",
    "Patient: So I had a cold two weeks ago and my ears still feel a little blocked sometimes. No pain, no discharge, hearing seems fine. Is that something to get checked?",
    "Caller: Hi, I'm calling about my husband. He has a mild sore throat, says it's like a 2 out of 10. No fever, no trouble swallowing. Started yesterday. He's been taking lozenges.",
    "Patient: I've had a small lump or something in my neck for a while. It doesn't hurt, no fever, no weight loss. I'm not sure if it's a lymph node from a cold I had. Just want to know if I should come in.",
    "Caller: My toddler has a runny nose and a little cough. No fever for three days. She's eating and playing normally. Pediatrician said it's viral but we have an ENT history so I'm just checking.",
    "Patient: I get recurring nosebleeds, maybe once a week. They stop within ten minutes with pressure. No blood thinners, no trauma. Just wondering if I need to be seen.",
    "Caller: I have mild ear discomfort, not really pain. No discharge, no hearing change. It's been four days. I was on a plane last week so maybe it's pressure. It's getting better.",
    "Patient: So I've had a scratchy throat and mild cough for about five days. No fever, no difficulty breathing. I've been resting and it's slowly improving. Should I schedule something?",
    "Caller: Hi, my mom has some hoarseness. She's 72. It's been a week. No trouble swallowing or breathing. She had a cold before that. Is this something we need to get checked soon?",
    "Patient: I have this ringing in my ear that comes and goes. It's been there for a few weeks. No hearing loss that I notice, no dizziness. Just annoying. Do I need an appointment?",
    "Caller: My child has a stuffy nose and mild sore throat. No fever for two days. Drinking fine, sleeping okay. Symptoms seem to be improving. Just want to confirm we don't need to come in.",
    "Patient: I've had a bit of pain when I swallow on one side for two days. It's mild. No fever, no breathing issues. I can eat and drink. Is that routine or should I come in sooner?",
    "Caller: I have earwax buildup I think. My ear feels blocked and my hearing is a bit muffled on one side. No pain, no discharge. I've tried the drops. Should I come in to get it cleaned?",
    "Patient: So I had an ear infection that was treated with antibiotics last month. It cleared but the ear still feels a little full sometimes. No pain now. Do I need a follow-up?",
    "Caller: Hi, I'm calling for myself. I have mild facial pressure and congestion. No fever, no tooth pain. It's been about a week. Same as I get every year. Routine check or can I wait?",
    "Patient: I've had a small sore inside my mouth, on the side of my tongue. It's been four days. No fever, eating fine. Just wondering if ENT is the right place or if I should wait.",
    "Caller: My son has a runny nose and sneezing. No fever, no ear tugging. He's 4. He's active and eating. We just wanted to know if we need an appointment or if this is just a cold.",
    "Patient: I have a mild sore throat and a little bit of neck stiffness. No fever, no trouble swallowing. Started two days ago. I'm drinking a lot of fluids. Getting better I think.",
    "Caller: I've had a persistent dry cough for two weeks. No fever, no shortness of breath. It's worse at night. No other symptoms. Do I need to be seen by ENT?",
    "Patient: So I have this sensation of something stuck in my throat when I swallow. It's been a few days. No real pain, no weight loss, eating fine. Just uncomfortable. Should I come in?",
    "Caller: Hi, I have mild ear pain on one side. It started yesterday. No discharge, no hearing loss. It's not severe. I took some ibuprofen and it's a bit better. Do I need to be seen?",
    "Patient: I've had a cold and now my voice is a bit rough. No trouble breathing or swallowing. It's been about five days. Improving slowly. Is this something that needs a visit?",
    "Caller: My wife has had a stuffy nose and mild headache for a week. No fever. She says her ears feel a bit full. No pain. She's been using a decongestant. Should she come in?",
    "Patient: I have a small scratchy feeling in my throat when I talk a lot. No fever, no difficulty swallowing. It's been three days. Resting my voice seems to help. Routine or urgent?",
    "Caller: My dad has some hearing difficulty in one ear. It's been gradual over months. No pain, no discharge, no dizziness. He's 68. Wanted to see if he needs an appointment.",
    "Patient: I've had mild sore throat and swollen glands in my neck for three days. No fever today. I can swallow fine. It's a bit better than yesterday. Should I schedule?",
    "Caller: Hi, I have post-nasal drip and a mild cough. No fever, no shortness of breath. It's been about ten days. I'm not getting worse. Just checking if I need to be seen.",
    "Patient: So I have this clicking in my ear when I swallow. No pain, no hearing loss. It's been a couple of weeks. It's annoying but not severe. Do I need an ENT visit?",
    "Caller: My daughter has a mild sore throat and runny nose. No fever for 48 hours. She's drinking and eating. Pediatrician said viral. We have ENT for tubes history. Just double-checking if we need to come in.",
    "Patient: I've had a bit of dizziness when I stand up quickly. No spinning, no hearing loss. It's been a few days. I had a cold last week. Is this ENT or something else?",
    "Caller: I have a mild sore throat on one side. No fever, no trouble swallowing. Started two days ago. I've been gargling salt water. It's tolerable. Do I need an appointment?",
    "Patient: I've had congestion and a mild headache for five days. No fever, no facial pain. Sinus pressure type. OTC helps a bit. Should I come in or keep managing at home?",
    "Caller: My child has had a runny nose for a week. No fever, no ear pain. He's in daycare so we expect colds. Just want to make sure we don't need to see ENT. He's fine otherwise.",
    "Patient: I have a small bump behind my ear. It's been there for a few weeks. Doesn't hurt, no fever. Maybe a lymph node. Should I get it checked?",
    "Caller: Hi, I've had a mild sore throat and fatigue for three days. No high fever, no difficulty swallowing. I think it's viral. It's not getting worse. Do I need to be seen?",
    "Patient: So I have this feeling of fullness in both ears. No pain, no hearing change. It started after a flight last week. It's slowly improving. Do I need an appointment?",
]

# More routine - expand with variations
def add_routine_variations():
    more = [
        "Nurse line: Caller states patient has sore throat two days, mild, no fever. Patient is drinking and eating. No dysphagia or breathing difficulty. Caller asking if appointment needed.",
        "Transcript: Patient says she has a cold with runny nose and mild cough. No fever for three days. She's improving. Wants to know if she needs to see ENT or just rest.",
        "Call: Dad calling for son age 7. Mild earache one day. No fever. Child is active. No discharge from ear. Asking if we need to bring him in.",
        "Caller: I have this throat clearing thing going on. No pain, no fever. Maybe allergies. It's been a week. Just annoying. Should I come in?",
        "Patient: I've had a mild sore throat that comes and goes. No fever, no trouble swallowing. It's been about a week. I think it might be reflux. Do I need to be seen?",
        "Transcript: Mother reports child with congestion and mild sore throat. No fever. Child drinking well. Symptoms improving. Requesting advice on whether to schedule.",
        "Caller: My ear has been itchy for a few days. No pain, no discharge, no hearing change. I use earbuds a lot. Should I get it checked?",
        "Patient: I have a mild sore throat and a little cough. No fever. Started four days ago. I'm feeling a bit better today. Just checking if I need an appointment.",
        "Call: Patient reports nasal congestion and mild facial pressure. No fever. Duration one week. No vision changes or severe headache. Asking for appointment recommendation.",
        "Caller: Hi, I've had a scratchy throat for three days. No fever, no difficulty swallowing. I've been using lozenges. It's mild. Do I need to come in?",
        "Triage note: Caller reports adult with mild sore throat 3 days. No fever. Eating and drinking normally. No dysphagia or breathing difficulty. Requesting routine appointment.",
        "Patient: I have a bit of ear pressure. No pain. Flew yesterday. No hearing loss. It's improving. Do I need to be seen?",
        "Call: Mother calling re child age 5. Runny nose and mild cough. No fever 48 hours. Child active, eating. Asking if ENT visit needed.",
        "Caller: I've had a mild sore throat for four days. No fever. I can swallow fine. It's actually getting better. Should I schedule or wait?",
        "Transcript: Patient with history of allergies presents with runny nose and sneezing. No fever. No ear or throat pain. Wants to know if needs appointment.",
        "Caller: My father has some hoarseness. He's 75. No trouble swallowing or breathing. It's been a week. He had a cold before. Routine or urgent?",
        "Patient: I have a small sore throat and feel a bit tired. No fever. Two days. I'm drinking fluids. Improving. Do I need to come in?",
        "Call: Child with mild ear discomfort. No fever. No discharge. One day. Parent asking if needs to be seen.",
        "Caller: I have post-nasal drip and a cough. No fever. It's been two weeks. Not getting worse. Just wondering if I need an ENT visit.",
        "Transcript: Adult with mild nasal congestion and sore throat. No fever. Symptoms improving over 3 days. Patient requesting advice on need for appointment.",
    ]
    return ROUTINE_TRANSCRIPTS + more


# --- SEMI-URGENT (moderate, worsening, fever, needs 24–48h) ---
SEMI_URGENT_TRANSCRIPTS = [
    "Caller: My husband has a really bad sore throat and fever of 101. It's been three days and it's getting worse. He can still swallow but it hurts. No trouble breathing. Should we come in soon?",
    "Patient: I have severe ear pain and there's some fluid coming out of my ear. I have a low-grade fever. It started two days ago. The pain is pretty bad. I need to be seen.",
    "Caller: My daughter has a high fever and a very sore throat. She's drinking but it hurts to swallow. She's 6. No stridor or breathing problems. It's been 48 hours and not improving. Can we get an appointment tomorrow?",
    "Patient: I've had sinus pain and pressure with a fever of 100.5 for four days. The headache is getting worse. I've been on decongestants. I think I need to be seen in the next day or two.",
    "Caller: Hi, I'm immunocompetent but I have a bad throat infection. Fever 102, severe pain when swallowing. I can breathe fine. It's been two days and it's not getting better. I need an appointment soon.",
    "Patient: So I have ear pain and hearing is muffled on one side. There's some discharge. I have a slight fever. It's been three days. The pain is moderate to severe. I'd like to be seen within 24 to 48 hours.",
    "Caller: My son has a sore throat and fever 101.5. He's had it for three days. He's drinking but not eating much. No difficulty breathing. Pediatrician said to follow up with ENT if not better. We need an appointment.",
    "Patient: I've had worsening throat pain for four days. Now I have a fever of 100.8. I can swallow but it's painful. No breathing issues. I'm worried it might be strep or something that needs treatment. Can I be seen soon?",
    "Caller: My wife has severe sinus pain and fever. She's had it for five days. The pain is in her face and teeth. No vision changes. She needs to be seen in the next day or two.",
    "Patient: I have a bad ear infection I think. Severe pain, some discharge, fever 99.8. It's been two days. I've tried OTC pain relief but it's not enough. I need an appointment.",
    "Caller: Hi, my dad has diabetes and he has a sore throat with fever 101. It's been two days. He can swallow. No breathing difficulty. We're worried because of his diabetes. Can he be seen within 24 to 48 hours?",
    "Patient: I've had a persistent sore throat with white spots. Fever 100.5. It's been three days. I can swallow with pain. No stridor. I need to be seen soon to rule out something serious.",
    "Caller: My child has ear pain and is pulling at his ear. He has a fever of 100.2. He's 3. He's drinking some. No breathing problems. It's been 36 hours. We need an appointment tomorrow.",
    "Patient: So I have a throat infection that's not improving. Fever 101.5, severe pain on swallowing. I'm on day four. I can breathe fine. I need to be seen in the next day or two.",
    "Caller: I have severe tonsil pain on one side. Fever 100.8. It's been three days. I can swallow liquids. No difficulty breathing. I think I might need antibiotics. Can I get an appointment soon?",
    "Patient: I've had ear pain and muffled hearing for four days. Now I have a low-grade fever. There's a little discharge. The pain is significant. I'd like to be seen within 24 to 48 hours.",
    "Caller: My mom has a bad cold and now she has ear pain and fever. She's 70. She can hear okay. No dizziness. She's had it for three days. We want her seen in the next day or two.",
    "Patient: I have worsening sore throat and fever 102. It's been 48 hours. I can swallow with pain. No breathing difficulty. I'm worried about strep or abscess. I need to be seen soon.",
    "Caller: Hi, my teenager has a severe sore throat and fever. He can swallow but it hurts a lot. No stridor. It's been three days. He's not improving. We need an appointment within 24 to 48 hours.",
    "Patient: So I have sinusitis I think. Facial pain, pressure, fever 100.4. It's been five days. I've tried OTC. It's not improving. I need to be seen in the next day or two.",
    "Caller: My husband has ear pain and discharge. He has a fever of 99.5. It's been two days. The pain is bad. He needs to be seen soon. Can we get an appointment?",
    "Patient: I've had a sore throat that's gotten worse. Now I have fever 101.2 and it hurts to swallow. I can breathe fine. Duration three days. I need an appointment within 24 to 48 hours.",
    "Caller: My daughter has a fever and really bad sore throat. She's 8. She's drinking but crying when she swallows. No breathing problems. It's been two days. We need to be seen soon.",
    "Patient: I have moderate to severe ear pain and low-grade fever. Some drainage from the ear. It's been three days. I need to be seen in the next day or two.",
    "Caller: Hi, I have a throat infection with fever 101. I can swallow but it's painful. No difficulty breathing. It's been 72 hours and not improving. I'd like an appointment soon.",
    "Patient: So I've had worsening throat pain with fever. Now 101.5. I can swallow liquids. No stridor. Duration four days. I need to be seen within 24 to 48 hours.",
    "Caller: My child has ear infection symptoms. Ear pain, fever 100, irritable. He's 2. No breathing issues. It's been 48 hours. We need an appointment tomorrow.",
    "Patient: I have severe sinus pain and fever 100.8. It's been four days. No vision changes. The pain is not controlled with OTC. I need to be seen in the next day or two.",
    "Caller: My wife has a bad sore throat and fever. She's diabetic. She can swallow. No breathing difficulty. It's been two days. We're worried. Can she be seen within 24 to 48 hours?",
    "Patient: I've had ear pain and discharge for three days. Now I have a fever. The pain is significant. I need an appointment soon.",
    "Caller: Hi, I have a peritonsillar swelling on one side. I have fever 100.5. I can swallow with pain. No breathing difficulty yet. It's been two days. I need to be seen soon.",
    "Patient: So I have a throat and ear infection. Fever 101, severe pain. I can breathe fine. It's been three days. Not improving. I need to be seen within 24 to 48 hours.",
    "Caller: My dad has COPD and he has a sore throat with fever 100.2. He's not short of breath from the throat. He can swallow. It's been two days. We want him seen in the next day or two.",
    "Patient: I have worsening sore throat, fever 101.8. I can swallow. No stridor. Four days duration. I need an appointment soon.",
    "Caller: My son has high fever and sore throat. He's 5. Drinking fluids. No breathing difficulty. It's been 48 hours. Pediatrician said to see ENT if not better. We need an appointment.",
    "Patient: I've had ear pain and fever for three days. There's discharge. The pain is moderate to severe. I need to be seen within 24 to 48 hours.",
    "Caller: I have severe tonsillitis symptoms. Fever 101.2, bad pain on swallowing. I can breathe fine. Three days. I need to be seen soon.",
    "Patient: So I have sinus infection with facial pain and fever 100.5. Five days. Not improving with OTC. I need to be seen in the next day or two.",
]

# --- URGENT (red flags: breathing, stridor, sudden hearing loss, severe dysphagia, immunocompromised + infection) ---
URGENT_TRANSCRIPTS = [
    "Caller: Oh my god, my son is having trouble breathing. He has a really bad sore throat and he's making a noise when he breathes. He's 4. He's scared. We're on the way to the ER but I'm calling to see what we should do.",
    "Patient: I can't breathe properly. My throat is so swollen I can barely get air. I have severe sore throat and I'm starting to panic. I need help now.",
    "Caller: My husband has severe throat pain and he can't swallow his saliva. He's drooling. He has a fever. He says his throat is closing. We need to be seen immediately.",
    "Patient: I lost hearing in my right ear suddenly this morning. Like completely. No trauma, no infection that I know of. I'm really worried. I need to be seen as soon as possible today.",
    "Caller: My daughter has a bad sore throat and she's making a weird high-pitched sound when she breathes. She's 6. She's really struggling. We're heading to the emergency room. I'm calling to let you know.",
    "Patient: I'm immunocompromised, I'm on chemo. I have a severe sore throat and fever of 103. It's gotten worse in the last 12 hours. I'm scared. I need to be seen urgently.",
    "Caller: My dad has sudden hearing loss in one ear. It happened yesterday. He's 72. No trauma. He says it's like someone turned off the sound. He needs to be seen today.",
    "Patient: I have severe dizziness. I can't stand or walk. The room is spinning. I've been vomiting. I don't have hearing loss. I need to be seen urgently. I think it's my ear.",
    "Caller: My child has a high fever and severe sore throat and he's having difficulty breathing. He's making a noise when he inhales. We're going to the ER. I'm calling from the car.",
    "Patient: I have HIV and I have a bad throat infection with fever 102.5. My throat is getting more swollen. I can still breathe but I'm worried. I need to be seen today.",
    "Caller: My wife has severe throat pain and she can't swallow. She's drooling. She has a fever. She says it feels like her throat is closing. We need immediate help.",
    "Patient: I had sudden complete hearing loss in my left ear last night. No warning. No ear pain. I need to be seen as soon as possible. I've read that time matters.",
    "Caller: Hi, my son is 3 and he has stridor. He has a bad croupy cough and his breathing is noisy. He's working hard to breathe. We're on the way to the ER. Just letting the office know.",
    "Patient: I'm on immunosuppressants for my transplant. I have a severe sore throat and fever 103.5. It's worsening. I need to be seen urgently.",
    "Caller: My husband has severe vertigo. He can't stand. He's vomiting. He says the room is spinning. No hearing loss that he's noticed. He needs to be seen today. It's bad.",
    "Patient: I have difficulty breathing. My throat is very swollen. I have severe sore throat and fever. I'm getting scared. I need to be seen immediately.",
    "Caller: My daughter has a sore throat and she's struggling to breathe. She's making a whistling sound when she breathes in. She's 7. We're going to the emergency room now.",
    "Patient: I'm immunocompromised and I have a throat infection with high fever. My throat is getting worse. I need urgent evaluation. I can still breathe but I'm worried.",
    "Caller: My dad had sudden hearing loss in one ear this morning. Complete. No trauma. He's 65. He needs to be seen as soon as possible today.",
    "Patient: I have severe throat pain and I can't swallow. I'm drooling. I have a fever. I think I might have an abscess. I need to be seen right away.",
    "Caller: My child has stridor and high fever. He has a bad sore throat. He's 5. He's working to breathe. We're heading to the ER. I'm calling to inform the clinic.",
    "Patient: I have sudden severe vertigo. I can't walk. I've been vomiting. The room is spinning. It started a few hours ago. I need to be seen urgently.",
    "Caller: My wife is immunocompromised and she has a severe throat infection with fever 103. She's getting worse. She needs to be seen today. She can still breathe but we're worried.",
    "Patient: I can't breathe properly. My throat is swollen shut. I have severe pain. I need to go to the ER or be seen immediately.",
    "Caller: My son has difficulty breathing and a bad sore throat. He's making a noise when he breathes. He's 4. We're on the way to the hospital. Please advise.",
    "Patient: I had sudden complete hearing loss in one ear. No trauma. It happened yesterday. I need to be seen as soon as possible. I've heard there's a time window for treatment.",
    "Caller: Hi, my husband has severe throat swelling and he's having trouble breathing. He's drooling because he can't swallow. He has a fever. We need help now.",
    "Patient: I'm on chemotherapy and I have a severe sore throat and fever 102.8. It's getting worse. I need urgent care. I can breathe but I'm very concerned.",
    "Caller: My daughter has stridor. She has croup and her breathing is getting worse. She's 2. We're going to the ER. I'm calling to notify.",
    "Patient: I have severe throat pain and I cannot swallow. I'm drooling. I have a fever. I think I have an abscess. I need to be seen immediately.",
    "Caller: My dad has sudden hearing loss. One ear. It happened last night. No trauma. He's 70. He needs to be seen today. We're very worried.",
    "Patient: I have severe vertigo. I cannot stand. I've been vomiting. The world is spinning. No hearing loss. I need to be seen urgently. I think it's labyrinthitis or something.",
    "Caller: My child has a bad sore throat and is having trouble breathing. He's making a high-pitched sound. He's 6. We're on the way to the ER.",
    "Patient: I'm immunocompromised and I have a worsening throat infection with high fever. I need to be seen today. I can still breathe but I'm deteriorating.",
    "Caller: My wife has severe throat pain and can't swallow. She's drooling. She has a fever. She says her throat is closing. We need to be seen immediately.",
    "Patient: I have sudden unilateral hearing loss. Complete. It happened this morning. No trauma. I need to be seen as soon as possible today.",
    "Caller: Hi, my son is struggling to breathe. He has a sore throat and he's making a weird noise when he breathes in. He's 5. We're going to the emergency room. Just wanted to let you know.",
    "Patient: I have severe throat swelling and difficulty breathing. I have a fever. I need to be seen right away. I'm scared.",
    "Caller: My husband had sudden hearing loss in one ear. Complete. No injury. He needs to be seen today. We've read that early treatment matters.",
    "Patient: I'm on immunosuppressants. I have a severe sore throat and fever 103. It's worsening. I need urgent evaluation. I can breathe but I'm very worried.",
]

def add_more_urgent():
    more = [
        "Transcript: Caller reports child with stridor and fever. Child has severe sore throat and difficulty breathing. En route to ER. Requesting guidance.",
        "Call: Patient states sudden complete hearing loss in one ear. No trauma. Occurred within last 24 hours. Patient requesting urgent appointment same day.",
        "Caller: Immunocompromised patient with severe throat infection. Fever 102.5. Worsening over 24 hours. Patient can swallow with difficulty. No stridor. Needs urgent evaluation.",
        "Patient: I cannot swallow. I am drooling. Severe throat pain and fever. I need to be seen immediately. I think my throat is closing.",
        "Transcript: Mother reports toddler with noisy breathing and severe sore throat. Stridor noted. Child in distress. Family en route to emergency department.",
        "Call: Patient with sudden onset severe vertigo. Unable to stand. Vomiting. No hearing loss. Requesting urgent same-day evaluation.",
        "Caller: My spouse has severe throat pain and cannot swallow saliva. Fever present. Patient says throat feels closed. Requesting immediate care.",
        "Patient: I have HIV and a bad throat infection. Fever 103. Getting worse. I need to be seen today. I can still breathe but I'm scared.",
        "Transcript: Sudden unilateral hearing loss. Complete. No trauma. Patient requests same-day appointment. States he has read that early treatment is critical.",
        "Caller: Child with stridor and high fever. Severe sore throat. Difficulty breathing. Family transporting to ER. Notifying ENT office.",
    ]
    return URGENT_TRANSCRIPTS + more


# --- LONG DIALOGUE-STYLE CALLS (AI + Patient, full triage flow) ---
def long_dialogues_routine():
    return [
        """AI: Hello! Thank you for calling the ENT clinic. Before we begin, may I please have your full name and date of birth?
Marcus: Yes, Marcus Webb, date of birth November 3, 1992.
AI: Thanks, Marcus. I see you're in our system. Can I confirm your phone number ending in 8822?
Marcus: Yes, that's right.
AI: How can I help you today? What symptoms are you calling about?
Marcus: I've had a mild sore throat for a couple of days. It's more like a scratch, nothing major.
AI: How long has this been going on?
Marcus: About two days—maybe three.
AI: Is it getting better, worse, or staying the same?
Marcus: It's actually getting a little better. I've been drinking tea and resting.
AI: On a scale of 1 to 10, how would you rate the discomfort?
Marcus: Maybe a 2 or 3. It's mild.
AI: Any fever, difficulty swallowing, or trouble breathing?
Marcus: No fever. I can swallow fine. No breathing issues at all.
AI: Any other symptoms—congestion, ear pain, dizziness?
Marcus: A little congestion, but no ear pain or dizziness.
AI: Any significant medical conditions we should know about, such as being immunocompromised?
Marcus: No, I'm healthy.
AI: Have you had something like this before?
Marcus: Yeah, I get a mild sore throat with colds sometimes.
AI: Thanks, Marcus. I have the information we need. A nurse will review and someone will call to schedule if we recommend a visit. Anything else today?
Marcus: No, that's it. Thanks.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Priya: Priya Sharma, date of birth July 22, 1988.
AI: Thank you, Priya. I've got you in our system. Can I confirm your contact number ending in 5544?
Priya: Yes, correct.
AI: What can we help you with today? What symptoms are you experiencing?
Priya: I have some congestion and a runny nose. It's been about four days. It's annoying but I'm not really sick.
AI: How long has this been going on?
Priya: Four days, maybe five.
AI: Are your symptoms getting better, worse, or staying the same?
Priya: About the same. Maybe a little better today.
AI: On a scale of 1 to 10, how severe are the symptoms?
Priya: Maybe a 3. It's more annoying than painful.
AI: Any fever, ear pain, or changes in hearing?
Priya: No fever. No ear pain. Hearing is fine.
AI: Any dizziness, vision changes, or throat pain that makes it hard to swallow?
Priya: No, none of that.
AI: Any medical conditions we should know about—immunocompromised, diabetes, etc.?
Priya: No.
AI: Have you had similar symptoms before?
Priya: Yes, usually with seasonal allergies or a cold.
AI: Thank you. We'll have someone review and follow up about scheduling if needed. Anything else?
Priya: No, thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
David: David Kim, date of birth January 14, 1979.
AI: Thanks, David. I see you're in our system. Can I confirm your phone ending in 3399?
David: Yes.
AI: How can I help you today? What symptoms are you calling about?
David: I have a bit of ear pressure—like my ears are full. No real pain. I was on a plane a few days ago.
AI: How long has this been going on?
David: About four days since the flight.
AI: Is it getting better, worse, or staying the same?
David: It's getting better. Yesterday was worse.
AI: On a scale of 1 to 10, how bothersome is it?
David: Maybe a 3. It's more annoying than painful.
AI: Any ear pain, discharge, hearing loss, or dizziness?
David: No pain. No discharge. Hearing seems fine. No dizziness.
AI: Any fever or sore throat?
David: No.
AI: Any significant medical history we should know about?
David: No.
AI: Have you had ear pressure after flying before?
David: Yes, a few times. It usually clears up.
AI: Thanks, David. We'll review and someone will call if we recommend a visit. Anything else?
David: No, that's all. Thanks.""",
        """AI: Hello! Thank you for calling the ENT clinic. Before we begin, may I have your full name and date of birth?
Jennifer: Jennifer Torres, date of birth April 8, 1995.
AI: Thank you, Jennifer. I have you in our system. Can I confirm your number ending in 6677?
Jennifer: Yes.
AI: What brings you in today? What symptoms are you calling about?
Jennifer: I've had a hoarse voice for about three days. I had a cold last week and my voice is still rough.
AI: How long has the hoarseness been going on?
Jennifer: About three days.
AI: Is it getting better, worse, or staying the same?
Jennifer: It's improving a little. I've been resting my voice.
AI: On a scale of 1 to 10, how would you rate the problem?
Jennifer: Maybe a 4. I can talk but it's scratchy.
AI: Any trouble swallowing, breathing, or pain when you talk?
Jennifer: No. I can swallow and breathe fine. No pain.
AI: Any fever, ear pain, or other symptoms?
Jennifer: No fever. No ear pain.
AI: Any medical conditions we should know about?
Jennifer: No.
AI: Have you had voice issues after a cold before?
Jennifer: Yes, once or twice. It always got better.
AI: Thanks, Jennifer. We'll review and follow up. Anything else?
Jennifer: No, thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: I'm calling for my son. His name is Tyler Reed, date of birth September 11, 2019. He's five.
AI: Thanks. I have Tyler in our system. Can I confirm the phone number ending in 2211?
Caller: Yes.
AI: What symptoms is Tyler having?
Caller: He has a runny nose and a mild sore throat. It's been two days. No fever today or yesterday.
AI: How long have the symptoms been going on?
Caller: Two days.
AI: Is he getting better, worse, or about the same?
Caller: He's getting better. He said his throat feels better today.
AI: How would you rate his discomfort—mild, moderate, or severe?
Caller: Mild. He's playing and eating normally.
AI: Is he drinking and eating okay? Any trouble breathing or swallowing?
Caller: Yes, he's drinking and eating. No trouble breathing. No weird sounds when he breathes.
AI: Any ear pain, ear tugging, or fever?
Caller: No ear pain. No fever for 48 hours.
AI: Any significant medical history for Tyler?
Caller: No. He's generally healthy.
AI: Has he had colds like this before?
Caller: Yes, he's in daycare. We just wanted to check if he needs to be seen.
AI: Thank you. We'll review and call if we recommend an appointment. Anything else?
Caller: No, that's it. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Thomas: Thomas Wright, date of birth June 19, 1981.
AI: Thanks, Thomas. I have you in our system. Phone ending in 8899?
Thomas: Yes.
AI: What symptoms are you calling about?
Thomas: I have a mild sore throat and a little cough. It's been about four days. No fever. I'm feeling a bit better today.
AI: How long has this been going on?
Thomas: Four days.
AI: Getting better, worse, or the same?
Thomas: Better. Yesterday was worse. Today it's milder.
AI: On a scale of 1 to 10?
Thomas: Maybe a 2 or 3. It's mild.
AI: Any difficulty swallowing, breathing, fever, or ear pain?
Thomas: No. I can swallow and breathe fine. No fever. No ear pain.
AI: Any other medical conditions?
Thomas: No.
AI: Have you had similar symptoms before?
Thomas: Yes, with colds. It usually clears up.
AI: Thanks, Thomas. We'll review and someone will call if we recommend a visit. Anything else?
Thomas: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Amanda: Amanda Foster, date of birth November 7, 1993.
AI: Thank you, Amanda. I see you in our system. Can I confirm your number ending in 2233?
Amanda: Yes.
AI: What can we help you with today?
Amanda: I have some ear discomfort—not really pain. My ear feels full. I was on a plane five days ago. No discharge, no hearing loss.
AI: How long has this been going on?
Amanda: Since the flight—five days.
AI: Is it getting better, worse, or the same?
Amanda: It's getting better. Slowly. Yesterday was a bit better than the day before.
AI: On a scale of 1 to 10, how bothersome?
Amanda: Maybe a 3. It's annoying but not painful.
AI: Any ear pain, discharge, hearing loss, dizziness?
Amanda: No pain. No discharge. Hearing seems fine. No dizziness.
AI: Any fever or sore throat?
Amanda: No.
AI: Any medical conditions we should know about?
Amanda: No.
AI: Have you had ear pressure after flying before?
Amanda: Yes. It usually goes away in a week or so.
AI: Thanks, Amanda. We'll review and follow up if needed. Anything else?
Amanda: No. Thank you.""",
    ]


def long_dialogues_semi_urgent():
    return [
        """AI: Hello! Thank you for calling the ENT clinic. Before we begin, may I please have your full name and date of birth?
Sarah: Sure—this is Sarah Chen, date of birth March 15, 1985.
AI: Thanks, Sarah. I see you're already in our system. Can I quickly confirm your contact information ending in 0101?
Sarah: Yes, that's correct.
AI: Great. How can I help you today? What symptoms are you calling about?
Sarah: I've been having this pressure and pain on the left side of my face, around my cheek and eye.
AI: Thank you. How long has this been going on?
Sarah: About a week now—maybe eight days.
AI: Has the pain been getting better, worse, or staying about the same?
Sarah: It's getting a little worse. Yesterday it felt more intense when I bent forward.
AI: Understood. On a scale from 1 to 10, how severe is the pain right now?
Sarah: Maybe a 6, sometimes a 7.
AI: Thank you. Have you noticed anything that makes the symptoms worse or better?
Sarah: Bending over makes the pressure worse. A warm compress helped a little.
AI: Have you had any other symptoms such as fever, nasal discharge, congestion, dizziness, or changes in vision?
Sarah: I've had some congestion and a yellowish nasal discharge. No fever. Vision is fine.
AI: Any significant medical conditions we should know about, such as being immunocompromised?
Sarah: No, I'm generally healthy.
AI: Thank you. One last question: Have you experienced this type of sinus issue before?
Sarah: Yes, I had something similar a couple years ago but not this bad.
AI: Thanks, Sarah. I've collected the information we need. A summary of your symptoms will be sent to clinic staff, and someone will follow up to schedule the appropriate next steps. Do you have any other concerns today?
Sarah: No, that's everything. Thank you.
AI: You're welcome. We'll be in touch shortly.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
James: James O'Brien, date of birth October 29, 1976.
AI: Thanks, James. I have you in our system. Can I confirm your number ending in 4455?
James: Yes.
AI: What symptoms are you calling about today?
James: I have a really bad sore throat and a fever. It's been three days and it's getting worse.
AI: How long has this been going on?
James: Three days. Maybe a bit more.
AI: Is it getting better, worse, or staying the same?
James: Worse. It hurt to swallow yesterday but today it's even more painful.
AI: On a scale of 1 to 10, how severe is the throat pain?
James: Maybe a 7 or 8. It's bad.
AI: Are you able to swallow—solids, liquids, or both?
James: I can swallow liquids. Solids are really painful. I'm mostly drinking.
AI: Any difficulty breathing, noisy breathing, or feeling like your throat is closing?
James: No. I can breathe fine. No weird sounds.
AI: What's your fever been? Any other symptoms—congestion, ear pain?
James: My fever was 101 this morning. I have some congestion. No ear pain.
AI: Any medical conditions we should know about—diabetes, immunocompromised?
James: I have type 2 diabetes. It's controlled.
AI: Have you had a severe sore throat like this before?
James: Not this bad. I've had strep before but it wasn't this painful.
AI: Thanks, James. We'll have a nurse review this. Given your symptoms and diabetes, someone will call you soon to schedule. You may need to be seen within 24 to 48 hours. Anything else?
James: No. I just need to be seen soon. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Maria: Maria Santos, date of birth December 5, 1990.
AI: Thank you, Maria. I see you in our system. Phone ending in 7788?
Maria: Yes.
AI: What can we help you with today? What symptoms are you experiencing?
Maria: I have bad ear pain and there's fluid coming out of my right ear. I also have a low-grade fever.
AI: How long has this been going on?
Maria: Two days. The pain started two days ago. The discharge started yesterday.
AI: Is the pain getting better, worse, or staying the same?
Maria: It's staying bad. Maybe a little worse today.
AI: On a scale of 1 to 10, how severe is the ear pain?
Maria: Maybe a 7. It's pretty bad. I've been taking ibuprofen.
AI: Any hearing loss, dizziness, or ringing in the ear?
Maria: My hearing is a bit muffled on that side. No dizziness. A little ringing.
AI: Any sore throat or difficulty swallowing?
Maria: No. Just the ear.
AI: What has your fever been?
Maria: Around 99.5 to 100. I checked this morning.
AI: Any medical conditions we should know about?
Maria: No.
AI: Have you had an ear infection before?
Maria: Yes, as a kid. Not in a long time.
AI: Thanks, Maria. We'll have someone review this. You likely need to be seen within 24 to 48 hours. We'll call to schedule. Anything else?
Maria: No. I just want to get in soon. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: I'm calling for my daughter. Emma Wilson, date of birth June 3, 2016. She's eight.
AI: Thanks. I have Emma in our system. Can I confirm your number ending in 9900?
Caller: Yes.
AI: What symptoms is Emma having?
Caller: She has a high fever and a really sore throat. She's drinking but it hurts to swallow. She's been crying.
AI: How long has this been going on?
Caller: About 48 hours. Maybe a little more.
AI: Is she getting better, worse, or about the same?
Caller: Not really better. The fever was 101.5 this morning. Same as yesterday.
AI: On a scale of 1 to 10, how would you rate her pain or discomfort?
Caller: She said 7 or 8. She doesn't want to eat.
AI: Is she able to swallow liquids? Any trouble breathing or noisy breathing?
Caller: She's drinking water and juice. No trouble breathing. No weird sounds when she breathes.
AI: Any ear pain, ear tugging, or discharge?
Caller: No ear stuff. Just the throat and fever.
AI: Any medical history we should know about for Emma?
Caller: No. She's healthy.
AI: Has she had strep or severe sore throat before?
Caller: She had strep once last year. This feels similar but we wanted to get her in.
AI: Thank you. We'll have a nurse review. We'll likely need to see her within 24 to 48 hours. Someone will call to schedule. Anything else?
Caller: No. We just want an appointment as soon as possible. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Robert: Robert Lee, date of birth August 17, 1965.
AI: Thanks, Robert. I have you in our system. Phone ending in 3344?
Robert: Yes.
AI: What symptoms are you calling about?
Robert: I have severe sinus pain and pressure. I've had it for five days. I have a fever too. It's not getting better.
AI: How long have the symptoms been going on?
Robert: Five days. Maybe six.
AI: Are they getting better, worse, or staying the same?
Robert: Worse. The headache and facial pain are worse than a few days ago.
AI: On a scale of 1 to 10, how severe is the pain?
Robert: Maybe a 7. The pressure in my face is really bad. Especially when I bend over.
AI: What has your fever been? Any nasal discharge, congestion?
Robert: Fever around 100.5. I have thick yellowish discharge and congestion.
AI: Any vision changes, severe headache, or neck stiffness?
Robert: No vision changes. Headache yes. No neck stiffness.
AI: Any medical conditions—COPD, immunocompromised?
Robert: I have COPD. It's been okay. I'm not more short of breath than usual.
AI: Have you had sinus infections before?
Robert: Yes. Usually I need antibiotics. I've been using decongestants but it's not helping enough.
AI: Thanks, Robert. Given your symptoms and COPD, we'll have someone review and call to schedule. You may need to be seen within 24 to 48 hours. Anything else?
Robert: No. I need to get in soon. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Karen: Karen Martinez, date of birth September 2, 1974.
AI: Thanks, Karen. I have you in our system. Phone ending in 6677?
Karen: Yes.
AI: What symptoms are you calling about?
Karen: I have a bad sore throat and white spots. I have a fever—100.5. It's been three days. I can swallow but it hurts a lot.
AI: How long has this been going on?
Karen: Three days.
AI: Getting better, worse, or the same?
Karen: Worse. Today it's more painful than yesterday.
AI: On a scale of 1 to 10?
Karen: 7. It's pretty bad.
AI: Are you able to swallow liquids and solids? Any difficulty breathing?
Karen: I can swallow. It hurts. No trouble breathing. No weird sounds.
AI: Any congestion, ear pain, or other symptoms?
Karen: A little congestion. No ear pain.
AI: Any medical conditions—immunocompromised, diabetes?
Karen: No. I'm healthy.
AI: Have you had strep or severe sore throat before?
Karen: I had strep once. This feels similar. I need to be seen in the next day or two.
AI: Thanks, Karen. We'll have someone review and call to schedule. You may need to be seen within 24 to 48 hours. Anything else?
Karen: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: I'm calling for my son. His name is Noah Davis, date of birth April 22, 2018. He's six. He has ear pain and a fever. It's been two days. The pain is bad.
AI: Thanks. I have Noah in our system. Can I confirm your number ending in 5544?
Caller: Yes.
AI: What symptoms is Noah having?
Caller: Ear pain on the right side. Fever—100.2 this morning. He's irritable. He's had a little discharge from the ear since yesterday.
AI: How long has this been going on?
Caller: Two days. Maybe 36 hours for the discharge.
AI: Is he getting better, worse, or the same?
Caller: The same. Not really improving. He's still in pain.
AI: How would you rate his pain—mild, moderate, or severe?
Caller: Moderate to severe. He's been crying. We've given him ibuprofen.
AI: Is he drinking and eating? Any trouble breathing?
Caller: He's drinking some. Not eating much. No trouble breathing. No weird breathing sounds.
AI: Any sore throat or other symptoms?
Caller: No. Just the ear.
AI: Any medical history for Noah?
Caller: No. He's healthy.
AI: Has he had ear infections before?
Caller: Yes, once or twice. We need to get him in within a day or two.
AI: Thank you. We'll have a nurse review and call to schedule. We'll likely get him in within 24 to 48 hours. Anything else?
Caller: No. Thank you.""",
    ]


def long_dialogues_urgent():
    return [
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: This is urgent. My son is having trouble breathing. He has a really bad sore throat and he's making a noise when he breathes. He's four. We're on the way to the ER. I'm just calling to let you know.
AI: I understand. Can I get his name and date of birth quickly?
Caller: Jake Morrison, date of birth February 20, 2020.
AI: Is he able to speak or cry? Is the breathing very labored?
Caller: He's crying. He's making a high-pitched sound when he breathes in. He's really scared. We're five minutes from the ER.
AI: You're doing the right thing going to the ER. For our records: How long has the sore throat and breathing trouble been going on?
Caller: The sore throat started two days ago. The breathing noise started this morning. It got worse in the last hour.
AI: Any fever? Is he able to swallow?
Caller: He has a fever—101. He's not really swallowing. He's drooling a bit. He won't drink.
AI: Thank you. Please go straight to the ER. We'll note this in his chart. Call us back after he's been evaluated if you need a follow-up with ENT. Is there anything else I can help with right now?
Caller: No. We're almost there. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Linda: Linda Park, date of birth May 9, 1972. I need to be seen today. I had sudden hearing loss in my right ear.
AI: I'm sorry to hear that. When did the hearing loss occur?
Linda: This morning. I woke up and the right ear was completely dead. No sound at all.
AI: Was there any trauma, loud noise, or ear infection before this?
Linda: No. Nothing. I was fine yesterday. I went to bed and woke up and couldn't hear on that side.
AI: Any dizziness, ringing, or ear pain?
Linda: A little ringing. No dizziness. No pain.
AI: Have you had any recent cold or infection?
Linda: I had a mild cold last week. It was almost gone.
AI: Any other medical conditions we should know about?
Linda: No. I'm healthy. I read that sudden hearing loss needs to be treated quickly. I need an appointment today.
AI: You're correct—sudden hearing loss is treated as urgent. We'll have a nurse call you back within the hour to get you in today. Can I confirm your phone number ending in 1122?
Linda: Yes. Please have someone call as soon as possible.
AI: We will. Someone will call you shortly. Anything else?
Linda: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Michael: Michael Torres, date of birth November 21, 1982. I'm immunocompromised—I'm on chemotherapy. I have a severe sore throat and high fever. I'm getting worse.
AI: I'm sorry. How long have you had the sore throat and fever?
Michael: About 36 hours. My throat is really bad. Fever was 103 this morning.
AI: Are you able to swallow—liquids or saliva?
Michael: I can swallow with a lot of pain. I'm managing liquids. I'm not drooling.
AI: Any difficulty breathing or noisy breathing?
Michael: No. I can breathe. I'm just really worried because of my immune system.
AI: Have you contacted your oncologist?
Michael: Yes. They said to see ENT if I have throat or sinus issues. I need to be seen today.
AI: Given your immunocompromised status and symptoms, we will have someone call you within the hour to schedule you for today. Can I confirm your number ending in 5566?
Michael: Yes. Please. I'm really concerned.
AI: We'll get you in today. Someone will call shortly. Anything else?
Michael: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: My wife can't swallow. She has severe throat pain and she's drooling. She has a fever. She says her throat is closing. We need to come in now.
AI: Is she able to breathe okay right now? Is she in distress?
Caller: She's breathing but she's really struggling. She's drooling because she can't swallow. She's in a lot of pain.
AI: If she is having significant difficulty breathing, you should call 911 or go to the ER immediately. Can you tell me her name and date of birth?
Caller: Rachel Green, date of birth July 4, 1988. We're thinking about going to the ER. How soon can you see her?
AI: With those symptoms—severe throat pain, inability to swallow, drooling—she needs to be evaluated urgently. If you are not already on the way to the ER, I strongly recommend going. We can also try to get her in within the next hour if you prefer to come here. Which would you prefer?
Caller: We'll go to the ER to be safe. Can you note this in her chart?
AI: Yes. We'll document this. Please go to the ER. They can manage potential airway issues. Call us after she's been evaluated. Anything else?
Caller: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Patricia: Patricia Nguyen, date of birth March 30, 1958. I have severe vertigo. I can't stand or walk. I've been vomiting. The room is spinning. I need to be seen today.
AI: I'm sorry. When did this start?
Patricia: A few hours ago. I was fine this morning. Then suddenly everything started spinning. I've vomited three times.
AI: Any hearing loss, ear pain, or ringing in your ears?
Patricia: No. My hearing is fine. No ear pain. Maybe a little ringing.
AI: Any weakness, numbness, or trouble speaking?
Patricia: No. Just the spinning and vomiting. I can't stand without falling.
AI: Any recent cold or ear infection?
Patricia: I had a cold two weeks ago. It's gone.
AI: Any significant medical history?
Patricia: I have high blood pressure. It's controlled.
AI: This could be a vestibular issue that needs prompt evaluation. We'll have someone call you within the hour to get you in today. Can I confirm your number ending in 7788?
Patricia: Yes. I need to be seen. I can't function like this.
AI: We'll get you in. Someone will call shortly. Anything else?
Patricia: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: I'm calling for my dad. He had sudden hearing loss in one ear yesterday. His name is William Hart, date of birth January 12, 1955. He's 69. He needs to be seen as soon as possible.
AI: When did the hearing loss occur?
Caller: Yesterday afternoon. He said he was watching TV and suddenly he couldn't hear out of his left ear. Completely gone. No trauma, no loud noise.
AI: Any ear pain, discharge, dizziness, or ringing?
Caller: No pain. No discharge. A little ringing. No dizziness.
AI: Any recent illness or infection?
Caller: He had a mild cold a couple weeks ago. Otherwise fine.
AI: Sudden hearing loss is treated as urgent. We'll have someone call you within the hour to schedule him for today or first thing tomorrow. Can I confirm the best phone number?
Caller: Yes, it's ending in 4433. Please call as soon as you can. He's really worried. We read that treatment needs to happen quickly.
AI: You're correct. We'll call shortly. Anything else?
Caller: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Daniel: Daniel Brown, date of birth October 11, 1968. I'm on immunosuppressants for a kidney transplant. I have a severe sore throat and fever 102.8. It's getting worse. I need to be seen today.
AI: I'm sorry. How long have you had the sore throat and fever?
Daniel: About 24 hours. Maybe a bit more. It's gotten worse in the last 12 hours.
AI: Are you able to swallow—liquids and saliva?
Daniel: I can swallow. It's painful. I'm not drooling. I can breathe fine.
AI: Any difficulty breathing or noisy breathing?
Daniel: No. Breathing is fine.
AI: Any other symptoms—congestion, ear pain?
Daniel: Some congestion. No ear pain.
AI: Have you contacted your transplant team?
Daniel: Yes. They said to see ENT urgently given my immunosuppression. I need an appointment today.
AI: We'll have someone call you within the hour to schedule you for today. Can I confirm your number ending in 9988?
Daniel: Yes. Please. I'm worried.
AI: We'll get you in today. Someone will call shortly. Anything else?
Daniel: No. Thank you.""",
        """AI: Hello! Thank you for calling the ENT clinic. May I have your full name and date of birth?
Caller: My daughter is making a high-pitched noise when she breathes. She has a bad sore throat and fever. She's six. We're on the way to the emergency room. I'm calling to let you know. Her name is Sophia Adams, date of birth August 15, 2018.
AI: You're doing the right thing going to the ER. How long has she had the sore throat and the breathing noise?
Caller: Sore throat for two days. The breathing noise started this morning. It's gotten worse in the last hour.
AI: Is she able to swallow or drink? Any drooling?
Caller: She's not really drinking. She's drooling a little. She's really scared.
AI: We'll note this in her chart. Please go straight to the ER. Call us back after she's been evaluated if you need ENT follow-up. Anything else?
Caller: No. We're almost at the hospital. Thank you.""",
    ]


def main():
    routine = add_routine_variations()
    semi = SEMI_URGENT_TRANSCRIPTS
    urgent = add_more_urgent()
    long_routine = long_dialogues_routine()
    long_semi = long_dialogues_semi_urgent()
    long_urgent = long_dialogues_urgent()

    examples = []
    # Short transcripts (repeated for volume)
    for _ in range(32):
        for t in routine:
            examples.append({"instruction": INSTRUCTION, "input": t, "output": "routine"})
    for _ in range(24):
        for t in semi:
            examples.append({"instruction": INSTRUCTION, "input": t, "output": "semi-urgent"})
    for _ in range(20):
        for t in urgent:
            examples.append({"instruction": INSTRUCTION, "input": t, "output": "urgent"})
    # Long dialogue-style calls (AI + Patient full triage flow; repeated so model sees them often)
    for _ in range(15):
        for t in long_routine:
            examples.append({"instruction": INSTRUCTION, "input": t, "output": "routine"})
    for _ in range(15):
        for t in long_semi:
            examples.append({"instruction": INSTRUCTION, "input": t, "output": "semi-urgent"})
    for _ in range(15):
        for t in long_urgent:
            examples.append({"instruction": INSTRUCTION, "input": t, "output": "urgent"})

    random.seed(42)
    random.shuffle(examples)

    counts = {"routine": 0, "semi-urgent": 0, "urgent": 0}
    for e in examples:
        counts[e["output"]] += 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        for rec in examples:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(examples)} transcript examples to {OUT_FILE}")
    print(f"  routine: {counts['routine']}, semi-urgent: {counts['semi-urgent']}, urgent: {counts['urgent']}")


if __name__ == "__main__":
    main()
