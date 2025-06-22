import os
import numpy as np
#from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import srt
from collections import Counter

from datetime import timedelta
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
 
#weights: categorys + codes are weighted as 1:
#bias criteria criteria .5 weight 
#bias exclusive criteria -.5 weight 


# Initialize the sentiment analysis pipeline
def calculate_sentiment_score(text, sentiment_tokenizer=sentiment_tokenizer, sentiment_model=sentiment_model):
    inputs = sentiment_tokenizer(text, return_tensors='pt')
    outputs = sentiment_model(**inputs)
    scores = outputs.logits.detach().numpy()[0]
    # Convert the scores to probabilities
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    # The sentiment score is the probability of 'positive' minus the probability of 'negative'
    sentiment_score = probabilities[2] - probabilities[0]
    return sentiment_score

def apply_research_codes_to_sentences(srt_file, codes, max_codes_per_sentence=None, tokenizer=tokenizer, model=model,  weights=None, bias_critera_weight=.5, bias_critera_exclusion_weight=.5, similarity_threshold=1.35,distance_threshold=1.18, sentiment_threshold=2.85, skip_sentimental_matches=False, remove_instructor=True, code_instructor=False, coded_output_only=False, veto_stats=(0,0,0,0,0,0)):
    #vd,    vsen,    vsim, = veto_stats
    vd, vsen, vsim, total_codecount,total_sub_list_len, total_sents = veto_stats

    with open(srt_file, 'r', encoding='utf-8') as file:
        subtitle_generator = srt.parse(file.read())
        subtitle_list = list(subtitle_generator)
    max_codes_per_sentence = int(len(codes)*.33) if max_codes_per_sentence is None else max_codes_per_sentence
    max_codes_per_sentence = min(max_codes_per_sentence, 6)
    text_list = [sub.content if "Instructor:" not in sub.content else ":instruction:" 
                 for sub in subtitle_list ] if not code_instructor else [sub.content for sub in subtitle_list]
    min_length_to_code=0
    sentences = [sent if sent!=":instruction:" else " " for sent in text_list ]
    embeddings = {'sentences': [], 'codes':[], 'inclusive': [], 'exclusive': []}
    code_definitions=[code['category'] + ','.join(code['codes']) + code['description'] for code in codes] 
    codes_inclusive= [code['inclusive'] for code in codes]
    codes_exclusive = [code['exclusive'] for code in codes]
    codes_applied_list = [] 
    if weights is None:
            weights = np.ones(len(codes))
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        try:
            outputs = model(**inputs)
            embeddings['sentences'].append(np.mean(outputs.last_hidden_state.detach().numpy(), axis=1))
        except:
            outputs = None
            embeddings['sentences'].append(outputs)
            print("DID NOT EMBED", sentence)
    for code in code_definitions:
        inputs = tokenizer(code, return_tensors='pt')
        outputs = model(**inputs)
        embeddings['codes'].append(np.mean(outputs.last_hidden_state.detach().numpy(), axis=1))

    for code in codes_inclusive:
        inputs = tokenizer(code, return_tensors='pt')
        outputs = model(**inputs)
        embeddings['inclusive'].append(np.mean(outputs.last_hidden_state.detach().numpy(), axis=1))

    for code in codes_exclusive:
        inputs = tokenizer(code, return_tensors='pt')
        outputs = model(**inputs)
        embeddings['exclusive'].append(np.mean(outputs.last_hidden_state.detach().numpy(), axis=1))

    """
    Now that we have the embeddings we can calculate the cosine similarity scores and Euclidean distances
    for the inclusive and exclusive criteria of the codes.
    """
    # Calculate the cosine similarity scores and Euclidean distances for inclusive and exclusive criteria
    similarity_scores_code_definitions     = [[cosine_similarity(sent_emb, code_emb)       for code_emb in embeddings['codes']] for sent_emb in embeddings['sentences']]
    euclidean_distances_code_definitions   = [[cdist(sent_emb, code_emb, 'euclidean')      for code_emb in embeddings['codes']] for sent_emb in embeddings['sentences']]
    similarity_scores_inclusive_criteria   = [[cosine_similarity(sent_emb, inclusive_emb)  for inclusive_emb in embeddings['inclusive']] for sent_emb in embeddings['sentences']]
    similarity_scores_exclusive_criteria   = [[cosine_similarity(sent_emb, exclusive_emb)  for exclusive_emb in embeddings['exclusive']] for sent_emb in embeddings['sentences']]
    euclidean_distances_inclusive_criteria = [[cdist(sent_emb, inclusive_emb, 'euclidean') for inclusive_emb in embeddings['inclusive']] for sent_emb in embeddings['sentences']]
    euclidean_distances_exclusive_criteria = [[cdist(sent_emb, exclusive_emb, 'euclidean') for exclusive_emb in embeddings['exclusive']] for sent_emb in embeddings['sentences']]


    sentence_sentiment_scores = [(calculate_sentiment_score(sentence)) for sentence in sentences]
    code_sentiment_scores = [calculate_sentiment_score(str(code['category']) +str(" ,".join(code['codes']) + str(code['description'])+"\n including: "+str(code['inclusive']))) for code in codes]
    # Calculate the combined similarity score and Euclidean distance
    combined_similarity_scores = np.zeros((len(sentences), len(codes)))
    combined_euclidean_distances = np.zeros((len(sentences), len(codes)))
    sentiment_scores = np.zeros((len(sentences), len(codes)))

    
    for i in range(len(sentences)):
        for j in range(len(codes)):
            combined_similarity_scores[i, j]    = (weights[j] * bias_critera_weight * similarity_scores_inclusive_criteria[i][j]) 
            combined_similarity_scores[i, j]   += weights[j]  * similarity_scores_code_definitions[i][j]
            combined_similarity_scores[i, j]   -= (weights[j] * bias_critera_exclusion_weight) * similarity_scores_exclusive_criteria[i][j] 
            combined_euclidean_distances[i, j]  = weights[j]  * euclidean_distances_code_definitions[i][j] 
            combined_euclidean_distances[i, j] += (weights[j] * bias_critera_weight * euclidean_distances_inclusive_criteria[i][j])
            combined_euclidean_distances[i, j] -= (weights[j] * bias_critera_exclusion_weight * euclidean_distances_exclusive_criteria[i][j])

            sentiment_scores[i, j]              =  abs(sentence_sentiment_scores[i] - code_sentiment_scores[j])

    similarity_scores = ( combined_similarity_scores)
    euclidean_distance_scores =  (combined_euclidean_distances)
    coded_sentences = [[sent] for sent in sentences]

    #average_similarities = np.mean(similarity_scores, axis=0)
    #average_distances = np.mean(euclidean_distance_scores, axis=0)
    # Calculate the average sentiment difference for each code

    codecount = 0 
    veto_by_distance=0
    veto_by_similarity=0
    veto_by_sentiment = 0
    max_codes=max(1,int(max_codes_per_sentence/3))
    for i, sentence in enumerate(sentences):
     if ((not code_instructor and sentence !=":instruction:")) or code_instructor:
        nearest_codes = np.argsort(euclidean_distance_scores[i])[:max_codes]
        if max_codes_per_sentence>1:
            most_similiar_codes = np.argsort(similarity_scores[i])[:max_codes]
        else:
            most_similiar_codes = []

        # Get the indices of the codes with the smallest differences
        if max_codes_per_sentence>2 or not skip_sentimental_matches:
            closest_sentiment = np.argsort(sentiment_scores[i])[:max_codes]
        else:
            closest_sentiment = []

        # Combine the lists of codes to consider for this sentence
        codes_to_consider = list(set(nearest_codes) | set(most_similiar_codes) | set(closest_sentiment))
        #print("\n\n\n CODE SENTENCE NUM:\n", i, "\n", sentence)
        for code in codes_to_consider:
          if code not in coded_sentences[i]:

            average_similarity = max(np.mean(similarity_scores[:, code]), np.mean(similarity_scores[:, :]))
            # Calculate the average sentiment and distance scores for this code
            average_sentiment_differences = max(np.mean(sentiment_scores[:, code]),np.mean(sentiment_scores[:, :]))
            average_distance = max(np.mean(euclidean_distance_scores[:, code]), np.mean(euclidean_distance_scores[:, :]))
            # Apply the code if the sentence's similarity score is above the average similarity score for this code
            # and the sentence's distance score is below the average distance score for this code
            #if abs(sentence_sentiment_scores[i] - code_sentiment_scores[code]) < average_sentiment_differences / sentiment_threshold:
            
            if sentiment_scores[i, code] < average_sentiment_differences / sentiment_threshold:
                if similarity_scores[i, code] > average_similarity * similarity_threshold:
                    if euclidean_distance_scores[i, code]  < average_distance /  distance_threshold:
                        coded_sentences[i].append(((codes[code]['category'])))
                        codecount+=1
                        codes_applied_list.append(codes[code]['category'])
                        print(f"\n CODED: \n {sentence}\n{codes[code]['category']} \n")
                        print(f"sentence num: {i} \n contins length:",len(coded_sentences[i]))
                        print("Distance: ", euclidean_distance_scores[i, code],"< ",average_distance /  distance_threshold, "weighted from avg", average_distance)
                        print("Similarity: ", similarity_scores[i, code],">",average_similarity * similarity_threshold, "weighted from average", average_similarity)
                        print("Sentiment: ",(abs(sentence_sentiment_scores[i] - code_sentiment_scores[code])),"< ", average_sentiment_differences / sentiment_threshold, "weighted from average", average_sentiment_differences)
                    else: #veto for distance
                        veto_by_distance+=1
                        #print("VETOED BY DISTANCE", codes[code]['category'], "#", veto_by_distance)#, sentence)
                        #print("Distance scores:", euclidean_distance_scores[i, code])
                        #print(" < average_distance /  distance_threshold",average_distance /  distance_threshold)
                else: #veto for similarity
                    veto_by_similarity+=1
                    #print("VETOED BY SIMILARITY", codes[code]['category'], "#", veto_by_similarity)#, code, sentence)
                    #print("Similarity scores:", similarity_scores[i, code])
                    #print(" > average_similarity * similarity_threshold", average_similarity * similarity_threshold)
                    if not (euclidean_distance_scores[i, code]  < average_distance /  distance_threshold):
                        veto_by_distance+=1
                        #print("       Also to be vetoed by distance ", euclidean_distance_scores[i, code], "d < t ",average_distance /  distance_threshold )
                        #print("\n #",veto_by_distance, "\n\n")
            else: #veto by sentiment
                veto_by_sentiment+=1
                #print("VETOED BY SENTIMENT", codes[code]['category'], "#", veto_by_sentiment)
                #print("Sentiment scores:", abs(sentence_sentiment_scores[i] - code_sentiment_scores[code]))
                #print(" < average_sentiment_differences / sentiment_threshold",average_sentiment_differences / sentiment_threshold)
                if not (euclidean_distance_scores[i, code]  < average_distance /  distance_threshold):
                    veto_by_distance+=1
                 #   print("       Also to be vetoed by distance ", euclidean_distance_scores[i, code], "d < t",average_distance /  distance_threshold, "#",veto_by_distance, "\n\n")
                if not (similarity_scores[i, code] > average_similarity * similarity_threshold):
                    veto_by_similarity+=1
                  #  print("       Also to be vetoed by similarity", similarity_scores[i, code],"s > t ",average_similarity * similarity_threshold,"\n #",veto_by_similarity, "\n\n")
    applied_codes=coded_sentences
    #print(applied_codes)
    for i, sub in enumerate(subtitle_list):
         if not remove_instructor and not code_instructor:
             subtitle_list[i].content = "\n ~ == ".join(applied_codes[i]) if ":instruction:" not in applied_codes[i] else subtitle_list[i].content
         elif remove_instructor:
             subtitle_list[i].content = "\n ~ == ".join(applied_codes[i])  if (":instruction:" or "Instructor:") not in applied_codes[i] else " ... "
         else:
             assert(subtitle_list[i].content == subtitle_list[i].content)

    count_dict = Counter(codes_applied_list)
    codes_applied_ct = (count_dict.items())
    print(codes_applied_ct)
    #print((subtitle_list))
    coded_subtitle_list = [sub for sub in subtitle_list if "~ == " in sub.content]
    if coded_output_only:
        subtitle_list=coded_subtitle_list
    coded_srt=(srt.compose(subtitle_list))

    print("\n\n\n STATS: \n vetos by distance", veto_by_distance)
    print("\nvetos by sentiment", veto_by_sentiment)
    print("\nvetos by similarity", veto_by_similarity)
    print(f"\n Total codes applied: #{codecount}\n\n")
    print("Length of total in sentence list:", len(text_list))
    textlen = len([text for text in text_list if "Instructor:" and "..." and ":instruction:" not in text])
    print("\nLength without instructor:", (textlen))
    print("\nNum coded:",len(coded_subtitle_list))
    #print(coded_srt)
    vd += veto_by_distance
    vsen += veto_by_sentiment
    vsim += veto_by_similarity
    total_codecount += codecount
    total_sub_list_len += len(coded_subtitle_list)
    total_sents += (textlen)
    veto_stats=vd, vsen, vsim, total_codecount,total_sub_list_len, total_sents 
    return coded_srt, veto_stats, codes_applied_list
def standardize_scores(combined_scores):
    # Calculate the mean and standard deviation of the flattened array
    mean_score = np.mean(combined_scores)
    std_score = np.std(combined_scores)

    # Standardize the scores
    standardized_scores = (combined_scores - mean_score) / std_score

    return standardized_scores

def normalize_scores(combined_scores):
    # Flatten the array
    flattened_scores = combined_scores.flatten()
    min_score = min(flattened_scores)
    max_score = max(flattened_scores)

    # Normalize the scores
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in flattened_scores]

    # Reshape the normalized scores to match the original shape of combined_scores
    normalized_scores = np.reshape(normalized_scores, combined_scores.shape)

    return normalized_scores
def concat_srt_files(file_paths, output_path="concat.srt"):
    """
    Concatenates a list of .srt files into one.
    
    :param file_paths: List of file paths to .srt files to concatenate
    :param output_path: The output file path for the concatenated .srt file
    :return: The output file path
    """
    all_subtitles = []
    total_duration = timedelta()

    # Process each .srt file
    for file_path in file_paths:
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                subtitles = list(srt.parse(f))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue
        # Update subtitle index and timing
        if all_subtitles:  # if not the first file
            for subtitle in subtitles:
                subtitle.index = subtitle.index + len(all_subtitles) if subtitle.index is not None else len(all_subtitles)
                subtitle.start += total_duration
                subtitle.end += total_duration
        # Extend the master subtitle list and update the total duration
        all_subtitles.extend(subtitles)
        total_duration += subtitles[-1].end
    # Write all subtitles to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(all_subtitles))
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
        return None
    print(output_path)
    return output_path

import os
from datetime import timedelta
import srt

def concat_srt_files(file_paths, output_path="concat.srt"):
    """
    Concatenates a list of .srt files into one.
    
    :param file_paths: List of file paths to .srt files to concatenate
    :param output_path: The output file path for the concatenated .srt file
    :return: The output file path
    """
    all_subtitles = []
    total_duration = timedelta()

    # Process each .srt file
    for file_path in file_paths:
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                subtitles = list(srt.parse(f))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

        # Update subtitle index and timing
        if all_subtitles:  # if not the first file
            for subtitle in subtitles:
                subtitle.index = subtitle.index + len(all_subtitles) if subtitle.index is not None else len(all_subtitles)
                subtitle.start += total_duration
                subtitle.end += total_duration

        # Extend the master subtitle list and update the total duration
        all_subtitles.extend(subtitles)
        total_duration += subtitles[-1].end

    # Write all subtitles to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(all_subtitles))
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
        return None
    return output_path











def get_codebook(which=None):
    
    phenomenology_codebook=[
    #affective_domain = [
        {
            'category': 'Affective Flattening, Emotional Detachment, or Alexithymia',
            'description': 'A narrowed or diminished affective range, a lack of affective charge, and/or an inability to identify/distinguish emotions.',
            'codes': ['Diminished affective range', 'Emotional detachment', 'Alexithymia'],
            'inclusive': 'Descriptions of affective flattening indicate a narrowed affective range (the opposite of Affective Lability). Affective flattening may also reflect a lack of affect arising and/or the absence of a typical affective response to an experience that usually evokes one. Emotional detachment may be expressed as a disidentification with the emotional charge of experiences or circumstances, having emotions but not feeling them, and/or having emotions only as body sensations without a sense of affect. Descriptions of alexithymia indicate an inability to identify, understand, process, describe or distinguish emotions, or to distinguish between the somatic dimensions of emotions and other bodily sensations. Double-code with Mental Stillness when affective flattening or emotional detachment is described as also having a cognitive dimension, for instance in relation to descriptions of "equanimity" associated with progress in meditation. Double-code with Meta-Cognition when emotional detachment is associated with cognitive distancing.',
            'exclusive': 'Emotional detachment, flattening, or alexithymia that arises due to causal circumstances other than meditation practice. Distinguish from Depression, which is characterized by low and negative mood rather than an absence of affect or a diminished affective range. Differentiate from (or double-code with) Anhedonia (lack of interest or motivation), which is a Conative change.'
        },
        {
            'category': 'Affective Lability',
            'description': 'Rapid shifts in mood, mood swings, increased range of emotions, or strong, unwarranted reactions to situations.',
            'codes': ['Mood swings', 'Increased emotional range', 'Strong emotional reactions'],
            'inclusive': 'Being emotionally reactive or triggered, or strong shifts in mood. This may manifest as strong reactions to situations that would not have previously elicited such a reaction, or a greater range of both positive and negative emotions than ordinarily experienced. May be double-coded with Agitation or Irritability, or Rage, Anger, or Aggression, or conversely with Positive Affect.',
            'exclusive': 'Affective lability that is mentioned prior to meditation experience or that can be attributed to other causes. For unusually strong or unwarranted reactions to situations arising beyond the context of meditation, do not code instances in which meditation-related changes are not playing an influencing role. For moods that are primarily ruminative or depressive in nature, code as Depression.'
        },
        {
            'category': 'Agitation or Irritability',
            'description': 'An agitated or irritable mood, possibly accompanied by restlessness, distractibility, or uneasiness.',
            'codes': ['Agitation', 'Irritability', 'Restlessness'],
            'inclusive': 'Agitation and irritability are types of affective reactivity and may include descriptions of restlessness, distractibility, or uneasiness. Agitation and restlessness described with somatic language would still be coded here if an affective dimension is mentioned and would be double-coded with Somatic Energy or other Somatic domain categories in cases of intense psychomotor agitation. Other common double-codes include Increased Cognitive Processing, Perceptual Hypersensitivity, Affective Lability, Rage, or Fear or Anxiety.',
            'exclusive': 'Excitation or a restless mind not attributed to meditation practice. Distinguish from Rage, Anger, or Aggression, which is higher in intensity and degree and more likely to be associated with behavioral change, though these may be coded in proximity if agitation and irritability develop into anger, for instance.'
        },
        {
            'category': 'Change in Doubt, Faith, Trust, or Commitment',
            'description': 'Changes (increase or decrease) in doubt, faith, trust, or commitment in relation to religious doctrines, practices, goals, community, or in relation to oneself in any dimension of life, such as self-confidence.',
            'codes': ['Change in doubt', 'Change in faith', 'Change in trust', 'Change in commitment'],
            'inclusive': 'May include an increase or decrease of faith with respect to religious matters, such as particular teachings or worldviews. Also includes doubt in one\'s ability to attain enlightenment or to be a successful meditator or, conversely, descriptions of self-assurance and self-confidence in practice-related or other domains. May be double-coded with Change in Worldview, Change in Relationship to Meditation Community, or categories in the Conative domain, such as Change in Motivation or Goal.',
            'exclusive': 'Changes in doubt, faith, trust, or commitment prior to meditation or regarding issues unrelated to meditation or to one\'s identity as a meditation practitioner.'
        },
        {
            'category': 'Crying or Laughing',
            'description': 'Crying and laughing, and associated vocalizations.',
            'codes': ['Crying', 'Laughing', 'Associated vocalizations'],
            'inclusive': 'Crying or laughing—for any reason or none, associated with emotional content or not, and either during formal meditation or not—that is causally attributed to meditation. Include descriptions of associated vocalizations such as wailing, moaning, or others. May be double-coded with Affective Lability, Depression, or Positive Affect.',
            'exclusive': 'Crying or laughing that is not attributed (in whole or in part) to meditation practice. For instance, crying or laughing due to an environmental stimulus or social situation that is not impacted in some way by the effects of meditation.'
        },
        {
            'category': 'Depression, Dysphoria, or Grief',
            'description': 'Low, depressed, or sad moods, usually coupled with physical and behavioral manifestations that may or may not affect normal functioning. Depression includes feelings of intense sadness, emptiness, hopelessness, helplessness, guilt, or unworthiness. Dysphoria includes feelings of unease or dissatisfaction with life. Grief includes feelings of sorrow or longing associated with loss.',
            'codes': ['Depression', 'Dysphoria', 'Grief'],
            'inclusive': 'May also be characterized by a problems concentrating on the task at hand, forgetfulness, insomnia, loss of appetite, and/or general loss of energy, drive, or motivation. Thus, where relevant, double-code with Change in Executive Functioning, Self-Conscious Emotions, Anhedonia, Appetitive Changes or Sleep Changes. Grief in particular may be double-coded with Change in Relationship to Meditation Community, Change in Doubt, Faith, Trust, or Commitment or changes in the Conative Domain.',
            'exclusive': 'Feelings of lowered mood without explicit connection to meditation practice. Loss of meaning or doubt are coded under Change in Doubt. Differentiate from (or double-code with) Anhedonia (lack of interest or motivation), which is a Conative change.'
        },
        {
            'category': 'Empathic or Affiliative Changes',
            'description': 'Increased or decreased empathic connection to other people or to environmental stimuli.',
            'codes': ['Increased empathic connection', 'Decreased empathic connection'],
            'inclusive': 'Descriptions of increased empathic connection may include references to social emotions (love, attachment, union), valuing social communities or interpersonal relationships and, at the more extreme end, affective sensitivity and contagion (being impacted by or taking on the feelings of others). Descriptions of decreased empathic connection may include references to indifference about or increased aversion to social situations or relationships, in which case consider double-coding with Affective Flattening. May be double-coded with categories in the Social Domain (if behavioral changes are also reported) and/or Change in Self-Other Boundaries (when the change in boundaries also results in affective contagion).',
            'exclusive': 'Empathic or affiliative changes not attributed to or exaggerated by meditation practice. Affective contagion (reports of taking on the emotions of others) may need to be differentiated from or double-coded with both Perceptual Hypersensitivity as well as Change in Self-Other Boundaries, a category in the Sense of Self Domain.'
        },
        {
            'category': 'Fear, Anxiety, Panic, or Paranoia',
            'description': 'Feelings of fright or distress--with or without an external referent--and their corresponding physiological and behavior responses.',
            'codes': ['Fear', 'Anxiety', 'Panic', 'Paranoia'],
            'inclusive': 'Some feeling of fright or distress--ranging from anxiety to fear, terror, panic, or paranoia--usually stimulated by a phenomenon that the subject is unused to, whether an external stimulus or a dimension of their meditation experience. Fear should be coded when it is reported as a response to other meditation-related changes in this codebook. However, fear or terror may also appear as a strong feeling on its own, without known or specified content or cause. Also includes intense negative emotions such as panic or paranoia and their behavioral changes that emerge from unpleasant experiences attributed wholly or in part as effects of meditation. Consider double-coding with Delusional, Irrational, or Paranormal Beliefs when paranoia influences thought content.',
            'exclusive': 'Fear, anxiety, panic, or paranoia prior to meditation-related experiences, arising outside of the context of meditation, or arising due to another identifiable cause that does not become amplified or exaggerated by meditation. General references to being afraid of/about something not accompanied by a clear affective phenomenology should not be included.'
        },
        {
            'category': 'Positive Affect',
            'description': 'A state of positive or elevated mood or energy level, ranging on a continuum from low to high arousal.',
            'codes': ['Positive affect'],
            'inclusive': 'Positive affect attributed to meditation practice. Possible descriptions of positive feelings ranging from low to high levels of arousal include: peace, joy, love, gratitude, happiness, awe, wonder, excitation, enthusiasm, effusiveness, bliss, euphoria, ecstasy, rapture, grandeur, grandiosity, mania, or others. At the high arousal level, particularly with mania, positive affect may be accompanied by intense productivity or insight, in which case a double-code with Change in Worldview may be warranted. Mania may also be reported in conjunction with delusions of grandeur (or other delusions), or impairment in rational thinking such that one acts with overconfidence or disregard for ones safety, finances, etc., in which cases double-coding with Increased Cognitive Processing or Delusional, Irrational, or Paranormal Beliefs may be warranted.',
            'exclusive': 'Positive affect stimulated by external circumstances without clear association with meditative practice. General references to an experience being positive that don\'t clearly reference an elevated mood or other criteria denoting an affective change.'
        },
        {
            'category': 'Rage, Anger, or Aggression',
            'description': 'Feelings of intense displeasure or a retaliatory response, often caused by some adverse stimulus provoking an uncomfortable emotion.',
            'codes': ['Rage', 'Anger', 'Aggression'],
            'inclusive': 'Extreme feelings of displeasure, retaliation, anger or aggression, either in reaction to a stimulus or in the absence of a known specific stimulus. Rage, anger, and aggression may arise on their own as primary phenomenology, or they may be in response to meditation-related experiences or how such experiences were managed. A double-code with Affective Lability is warranted when the response is stronger than it would be under typical conditions. May also be double-coded with or coded in proximity to Agitation or Irritability. May lead to or correspond with various changes in the Social Domain.',
            'exclusive': 'Feelings of anger arising outside of a meditation-related context, in the past (before onset), or for reasons not causally linked to meditation. Differentiate Agitation or Irritability, which is marked primarily by reactivity, from Rage, Anger or Aggression, which is higher in intensity and degree and has different behavioral manifestations.'
        },
        {
            'category': 'Re-experiencing of Traumatic Memories or Affect Without Recollection',
            'description': "Either a recollection of some past traumatic event in the subject's life that may or may not have been repressed, and which is generally associated with strong emotions, or the upwelling of strong emotions without any corresponding memory, content, thought or other identifiable stimulus.",
            'codes': ['Traumatic memory recall', 'Emotional upwelling without recollection'],
            'inclusive': "Re-experiencing of traumatic memories or traumatic flashbacks include references to the recollection of some past traumatic event in the subject's life. These can also be coupled with the explicit mention of its relation to powerful emotional content such as grief, terror, or shame, in which case double-coding for those may also be warranted. Affect without recollection includes references to an experience of an unexpected, or sudden onset or upwelling of emotions without an identifiable stimulus or typical causal factor, such as a memory, a thought, or an evocative external circumstance. When re-experiencing is associated with specific locations in the body, double-code for (Release of) Pressure, Tension or Somatic Energy, accordingly.",
            'exclusive': "Reference to some past event in the subject's life but without mention of it being emotionally intense or traumatic. Reference to emotions or affective response which have clear causal relationship to thoughts, memories, or external circumstances and are not unexpected. Re-experiencing of trauma or emotion not attributed to meditation practices and/or arising in non-practice contexts would also be excluded."
        },
        {
            'category': 'Self-Conscious Emotions',
            'description': 'Emotions relating to one\'s sense of self and identity, as well as the awareness of reactions of others to oneself, whether real or imagined.',
            'codes': ['Guilt', 'Shame', 'Embarrassment', 'Envy', 'Pride'],
            'inclusive': 'Include descriptions of notable increases or decreases in self-conscious emotions (guilt, shame, embarrassment, envy or pride) caused or intensified by meditation practice. Code self-conscious emotions arising as primary phenomenology, as responses to other specific meditation-related experiences, or as responses to how those experiences were responded to. May be double-coded with categories in the Social Domain when impacting the practitioner\'s relationship to social, occupational, or meditation communities.',
            'exclusive': 'Self-conscious emotions arising prior to meditation or in a context in no way related to meditation-related experiences should not be coded. Changes related to self-confidence or self-esteem should be coded under Change in Doubt, Faith, Trust, or Commitment.'
        },
        #{
        #    'category': 'Suicidality',
        #    'description': 'Suicide, an affect-driven wanting to die, not wanting to continue with life, wishing to no longer being alive, thinking about dying and killing oneself',
        #    'codes': ['Passive suicidal ideation', 'Active suicidal ideation'],
        #    'inclusive': 'Includes thoughts of contemplating the ending of one\'s own life, whether they arise briefly and spontaneously or are a consequence of other affective states. Passive ideation includes wanting to die, thinking about no longer wanting to be alive or taking one\'s own life. Active ideation includes making specific plans for taking one\'s own life or making an attempt at taking one\'s own life.',
        #    'exclusive': "This is distinct from dealing with pain, coping, adapting to become comfortable with the discomfort or disability, or not being able to live the life that one had wanted due to external circumstances or illnesses."
        #},
            #]
    #cognitive_domain = [
        {
            'category': 'Change in Executive Functioning',
            'description': 'Either an inability to perform cognitive functions of decision making, concentration, and memory that the person used to be able to perform, or an enhanced ability in these domains of executive functioning.',
            'codes': ['Cognitive change', 'Memory impairment', 'Enhanced cognition'],
            'inclusive': 'A noticeable increase or decrease in any capacity to think, make decisions, memory recall, or the performance of other cognitive tasks. Such changes may have been noted by the subject themselves or by another observer (such as family, friends, coworkers, etc.). Diminished capacities in executive functioning also co-occur with extreme forms of Mental Stillness. When the person is prevented from normal functioning, work, or the ability to relate to others, double-code with Social or Occupational Impairment.',
            'exclusive': 'Any symptomology related to emotions, such as Depression, would not belong in this category, unless it also affected cognitive abilities, in which case double-coding would be appropriate. Disambiguate from Disintegration of Conceptual Meaning Structures, otherwise known as cognitive defusion.'
        },
        {
            'category': 'Change in Worldview',
            'description': 'A shift in ways of thinking about the nature of self or reality, including a change in understanding or confusion about the nature of self or reality.',
            'codes': ['Shift in self-perception', 'Change in understanding of reality'],
            'inclusive': 'A change in worldview as a result of contemplative practice, for instance a change in relationship to or understanding of Buddhist teachings, doctrines, or views. Change in worldview may arise during a content-driven practice, a content-minimal practice, or following a practice session so long as the influence of prior meditation is apparent. May have many associated effects, including changes in the Conative domain such as Change in Effort or Change in Motivation or Goal, changes in the Sense of Self Domain, or Scrupulosity. May also be double-coded with Change in Relationship to Meditation Community.',
            'exclusive': 'Change in worldview due to learning or an interpersonal influence not associated with meditation or the associated teachings that are implemented in practice. Reports of beliefs held independent of meditation-related changes or interpretations of prior phenomenology should not be coded. Ordinary confusion (not pertaining to views of self or reality) should be coded under Change in Executive Functioning. Causal attributions made to the effect of worldviews on meditation-experiences would be coded as Influencing Factors.'
        },
        {
            'category': 'Clarity',
            'description': 'Reports of clarity or lucidity as a mental state, quality of attention, or quality of consciousness, in which there is a heightened cognition of relevant stimuli and a diminished interference from non-relevant stimuli.',
            'codes': ['Mental clarity', 'Lucidity', 'Heightened cognition'],
            'inclusive': 'Primarily described as a pervasive quality of mind, consciousness, or awareness or attention. Often associated with a figure-ground shift such that the practitioner attends more to the field of awareness rather than or in addition to discrete perceptual objects. May also be thought of as the opposite of the fogginess or dullness of Mental Stillness, in the sense of a clarity of cognitive or perceptual processing. May be reported in conjunction with Increased Cognitive Processing, Perceptual Hypersensitivity or other changes in the Perceptual Domain, in which case double-code accordingly.',
            'exclusive': 'The perception of an environmental clarity (such as space or air) that is not attributed to a change in the quality of awareness should not be coded. Brightening of the visual field should be primarily coded under Visual Lights. Other forms of Perceptual Hypersensitivity may coarise with Clarity; if the change is described exclusively in perceptual terms with no cognitive dimension, code under the relevant Perceptual category instead. Some reports of expansiveness/spaciousness may be more appropriately coded under Change in Self-Other or Self-World Boundaries.'
        },
        {
            'category': 'Delusional, Irrational, or Paranormal Beliefs',
            'description': 'Holding with conviction and being influenced by one or more beliefs despite evidence to the contrary. Ascriptions of significance or meaning that are later disregarded or that might seem unusual or concerning to members of the practitioners broader culture or particular subculture. Attributions of paranormal agency, origin, or explanation for cognitive experiences.',
            'codes': ['Delusion', 'Irrational belief', 'Paranormal belief'],
            'inclusive': 'Reports of distortions of grandiosity, paranoia, or mania that lead to some form of misapprehending or no longer participating in consensual reality. Beliefs that were reported as being strongly held as true despite feedback from others to the contrary or based upon rational or empirical evidence. Or beliefs that were described by the subject in retrospect as delusional or irrational in nature. Subjects may or may not have "insight" into the delusional nature of their beliefs at the time they are describing, and subsequent attributions or reflections on their prior experience should be taken into account. Also include here beliefs that are reported as seeming unusual or concerning to an authority in their culture or subculture (such as a meditation teacher), a friend or family member, or a mental health professional. Includes beliefs that affect the structuring of perceptual, cognitive, affective, or interpersonal experiences. Includes magical thinking and paranormal beliefs such as communication with agents, paranormal powers, access to forms of knowledge that would not be verifiable by a second-person source or experiences of hypersalience (anomalous gnostic events). Paranormal beliefs may also be double-coded with Change in Worldview, especially in cases where paranormal experiences are valued as indicating a contact with a deeper, truer, and/or usually hidden reality. When non-consensual changes in perception cooccur with changes in belief, double-code for Hallucinations, Visions or Illusions.',
            'exclusive': 'Statements that appear far-fetched but are not demonstrated to have been viewed as delusional at the time or in retrospect by the practitioner, and were not be deemed delusional, irrational, or paranormal by a member of the practitioners subculture or broader culture (including Western psychiatry). Associated beliefs must have clear phenomenological referent or intersect with phenomenological reports. Casual post-hoc descriptions of prior events as "delusional" or "irrational" should not be coded unless the events themselves indicate clear cognitive changes meeting inclusion criteria. Delusions that manifest with a percept-like experience in the absence of perceptual input would be coded as Hallucinations, Visions, or Illusions. Should be disambiguated from the manic dimensions of Positive Affect, which may arise without delusional cognitive content.'
        },
        {
            'category': 'Disintegration of Conceptual Meaning Structures',
            'description': 'Percepts arise but are processed without their associated conceptual meaning, resulting in an inability to form conceptual representations of the perceptual world.',
            'codes': ['Loss of conceptual meaning', 'Difficulty forming representations'],
            'inclusive':'Reports of distortions of grandiosity, paranoia, or mania that lead to some form of misapprehending or no longer participating in consensual reality. Beliefs that were reported as being strongly held as true despite feedback from others to the contrary or based upon rational or empirical evidence. Or beliefs that were described by the subject in retrospect as delusional or irrational in nature. Subjects may or may not have "insight" into the delusional nature of their beliefs at the time they are describing, and subsequent attributions or reflections on their prior experience should be taken into account. Also include here beliefs that are reported as seeming unusual or concerning to an authority in their culture or subculture (such as a meditation teacher), a friend or family member, or a mental health professional. Includes beliefs that affect the structuring of perceptual, cognitive, affective, or interpersonal experiences. Includes magical thinking and paranormal beliefs such as communication with agents, paranormal powers, access to forms of knowledge that would not be verifiable by a second-person source or experiences of hypersalience (anomalous gnostic events). Paranormal beliefs may also be doublecoded with Change in Worldview, especially in cases where paranormal experiences are valued as indicating a contact with a deeper, truer, and/or usually hidden reality. When non-consensual changes in perception cooccur with changes in belief, double-code for Hallucinations, Visions or Illusions.',
            'exclusive':'Statements that appear far-fetched but are not demonstrated to have beenviewed as delusional at the time or in retrospect by the practitioner, and were not be deemed delusional, irrational, or paranormal by a member of the practitioner subculture or broader culture (including Western psychiatry). Associated beliefs must have clear phenomenological referent or intersect with phenomenological reports. Casual post-hoc descriptions of prior events as "delusional" or "irrational" should not becoded unless the events themselves indicate clear cognitive changes meeting inclusion criteria. Delusions that manifest with a percept-like experience in the absence of perceptual input would be coded as Hallucinations, Visions, or Illusions. Should be disambiguated from themanic dimensions of Positive Affect, which may arise without delusionalcognitive content. '
        },
        {
            'category': 'Mental Stillness',
            'description': 'An state in which there are few identifiable thoughts, a perceived absence of thought, or a poor awareness about the thinking process in general.',
            'codes': ['Absence of thought', 'Mental quiescence', 'Cognitive obscurity'],
            'inclusive': 'Mention of an absence of thought, whether positively or negatively valenced, intentionally sought or involuntarily experienced. Includes both target states of "calm abiding," or "mental quiescence" associated with concentration practice, as well as unexpected, prolonged, or undesirable states of "spacing out without mind wandering," "fogginess," or a general obscuration of cognitive processes. Lack of access to thought (inability to generate thought) is an extreme and often involuntary version of Mental Stillness.',
            'exclusive': 'In cases of mental fog, attempt to differentiate absence or thought from an a Change in Executive Functioning. Corresponding flattening of emotions or a lack of affect should be coded under Affective Flattening. Expected cycles of drowsiness or dullness should not be coded here.'
        },
        {
            'category': 'Meta-Cognition',
            'description': 'Meta-cognition, or meta-awareness, refers to an explicit knowledge of the content of thoughts or the thinking process. Meta-cognition can also entail a higher-order cognition of processes in other domains of experience, such affective, perceptual, somatic or sense of self.',
            'codes': ['Meta-awareness', 'Distancing from thoughts', 'Monitoring awareness'],
            'inclusive': 'Sustained meta-cognition or meta-awareness resulting from the practice of meditation. Includes reports of a "distancing" from thoughts, seeing thoughts as "just thoughts," or as seeing thoughts as transient events in the mind. Also includes the explicit reference of a "monitoring" awareness witnessing transient somatic, affective, or perceptual events.',
            'exclusive': 'A brief moment of meta-cognition, or meta-cognition not related to meditation or to particular views and values associated with meditation. Prolonged and intense forms of meta-cognition may develop into a Loss of Sense of Ownership over thoughts, emotions, or body sensations or a Loss of Agency over actions, at which point those codes would be more appropriate. Some forms of distancing from thoughts and the content of thoughts in which a cognitive event is no longer processed with associated meaning or significance may be better coded (or double-coded) as Disintegration of Conceptual Meaning Structures.'
        },
        {
            'category': 'Scrupulosity',
            'description': 'Obsessive thinking, specifically about moral or religious issues and behaviors.',
            'codes': ['Moral obsession', 'Religious obsession', 'Change in self-perception'],
            'inclusive': 'A change in self-perception or behavior in relation to religious teachings or moral values associated with contemplative practices. Also includes the exacerbation of existing dispositional tendency towards obsession over moral or religious issues when influenced by meditation or being in a meditation context. May often be precipitated by or result in Change in Worldview or Change in Effort, and may also be co-occurring with behavioral changes in the Social Domain.',
            'exclusive': 'Changes in self-perception and behavior coming from other religious contexts (such as a practitioner\'s former religious tradition), unless those tendencies are exaggerated by meditation. Only beliefs and associated behaviors related to morality or religious codes of conduct should be coded here. Disambiguate from Delusional, Irrational, or Paranormal Beliefs, Change in Doubt, Faith, Trust or Commitment, and Change in Worldview.'
        },
        {
            'category': 'Vivid Imagery',
            'description': 'An experience of intense, vivid and/or clear thoughts or mental images that arise involuntarily, or a report of an increased ability to visualize.',
            'codes': ['Vivid thoughts', 'Mental image clarity', 'Increased visualization'],
            'inclusive': 'Intense and potentially disconcerting thoughts that arise involuntarily and which are accompanied by an increased clarity or vividness. Vivid imagery or thought intrusions may also be accompanied by strong emotions such as Fear or bliss (Positive Affect).',
            'exclusive': 'Vivid fantastical images that are perceived as if external objects would be coded under Hallucinations. Vivid imagery that is associated with traumatic memories should primarily be coded as Re-experiencing of Traumatic Memories. Disambiguate from Clarity, which may be a coinciding quality of consciousness, but mentioned independent of vivid imagery. Disambiguate from Hallucinations, Visions, or Illusions which are percept-like changes in the Perceptual Domain.'
        },
    #]
    #perceptual_domain = [
        {
            'category': 'Derealization',
            'description': 'Surroundings are perceived as strange, unreal, or dreamlike, or perception is experienced as mediated by a fog, a lens, or some other filter that results in feeling cut off from the world.',
            'codes': ['Feeling of unreality', 'Strange surroundings', 'Dreamlike quality'],
            'inclusive': 'Mentioning a feeling of unreality, strangeness, unfamiliarity or dreamlike quality of surroundings, or reports of feeling cut off or alienated from the perceptual world. Includes confusion concerning whether one is dreaming or awake.',
            'exclusive': 'Changes in sense perception without the sense of feeling dreamlike, or unreal would be coded under the appropriate change in the Perceptual domain.'
        },
        {
            'category': 'Dissolution of Objects or Phenomena',
            'description': 'The dissolving or complete disappearance of visual objects or the entire visual field.',
            'codes': ['Dissolving objects', 'Disappearing visual field', 'Pixelating objects'],
            'inclusive': 'An experience in which parts of objects, entire objects, or the entire visual field is distorted to the point of seeming composed of points of scintillating light, pixelating, dissolving, or entirely disappearing.',
            'exclusive': 'Differentiate from Distortions in Time or Space, which does not include objects.'
        },
        {
            'category': 'Distortions in Time or Space',
            'description': 'An alteration in the subjective experience of spatial boundaries or relations and/or temporal causality or sequencing.',
            'codes': ['Distortion of time', 'Distortion of space', 'Distortion of personal history'],
            'inclusive': 'Abnormalities in one\'s perception of space and/or time. May refer to one\'s perception of distance, scale, time, causality, or personal history. Includes references to temporal gaps or absences.',
            'exclusive': 'Discussion of time and space that do not describe distortions but ordinary experiences of time and/or space.'
        },
        {
            'category': 'Hallucinations, Visions, or Illusions',
            'description': 'A hallucination is an experience of a percept that is not externally stimulated, is not shared by others, and is not taken to be veridical. When a visual percept that is not shared by others is taken to be veridical, it is a vision. An illusion involves a percept that is distorted, changed, or has features added to the raw percept.',
            'codes': ['Hallucinations', 'Visions', 'Illusions'],
            'inclusive': 'A hallucination is the experience of a percept in the absence of a corresponding sensory stimuli. Hallucinations can occur in any modality: visual, auditory, gustatory, olfactory, or proprioceptive.',
            'exclusive': 'Hallucinations or illusions not reported either in the context of meditation practice or in the context of post-meditation effects.'
        },
        {
            'category': 'Perceptual Hypersensitivity',
            'description': 'Unusual or atypical sensitivity to certain frequencies or volumes of sound (hyperacusis), to color (hyperchromia), to visual details, to light, to taste, to smell, or to embodiment.',
            'codes': ['Hyperacusis', 'Hyperchromia', 'Hyper-sensitivity'],
            'inclusive': 'Being extremely sensitive to sounds, light, colors, tactile sensations or other environmental stimuli. Also commonly described as difficulty tolerating everyday sounds, lights, or other sensory stimuli.',
            'exclusive': 'Sensitivity to sound, lights, colors, smells etc. that are not produced by externally present stimuli--that is, which are Hallucinations, Visions, or Illusions--would be coded there.'
        },
        {
            'category': 'Somatosensory Changes',
            'description': 'A change in proprioceptive information that affects one\'s perception of relative positions or dimensions of body parts or the body more generally.',
            'codes': ['Change in body schema', 'Distorted body perception', 'Change in body scale'],
            'inclusive': 'Increased proprioceptive information, or increased awareness of or sensitivity to the body schema. Distortions in the body schema resulting in a change in scale of body parts or the body in general, disappearance of body boundaries (such as feeling that arms are missing), inaccurate perception of position of body parts (such as feeling that legs as being twisted when they are not), or changes in body scale (such as body parts feeling larger or smaller or dissolving).',
            'exclusive': 'Mentioning the body in a way unrelated to one\'s perception of it.'
        },
    #]
    #sense_of_self_domain = [
        {
            'category': 'Change in Self-Other or Self-World Boundaries',
            'description': 'Expansion beyond or distortions in the typical sense of where the boundaries between self and other or self and world are delineated.',
            'codes': ['Expanded boundaries', 'Dissolved boundaries', 'Altered self-other boundaries'],
            'inclusive': "Includes references to being expanded beyond one's body schema, or the sense of boundaries between self and other or self and world being dissolved. May also be described in terms of merging with, being porous or permeable to either the world or other people. Alteration of self-other boundaries includes experiencing other people's mind states or emotions, in which case double-coding with Empathic or Affiliative Changes may be appropriate. References to awareness as spacious, expansive, centerless, non-local, all-pervading, or non-dual with the world of experience would be coded here, as would references to a sense of unity or oneness with nature, the environment, the world, or the universe. Conversely, feeling more separated from the world or feelings as if distant from it could also be coded here. Expansiveness may have associated phenomenology from other domains, such as increases in Clarity, Positive Affect, or Affective Lability.",
            'exclusive': "Does not include reports of cessation of consciousness or a sense of 'not being there.' Be careful to disambiguate with Change in Sense of Embodiment for reports where the sense of self is displaced to an atypical location within or in relation to the body schema. Hypersensitivity to others' emotions without reference to a change in boundaries should be coded as Empathic or Affiliative Changes. Hypersensitivity to the body schema without corresponding alterations in boundaries between self and other or self and world would be coded as Somatosensory Changes. Feeling more distant from others or from the world should be coded as Derealization when this is attributed to the visual sense of a filter or opaque medium, or to a loss of sense of reality."
        },
        {
            'category': 'Change in Narrative Self',
            'description': 'A report of a change in how the practitioner conceives of himself or herself as a person. Or, a change in the content of or their perspective on their story or personal identity.',
            'codes': ['Changed self-perception', 'Change in personal story'],
            'inclusive': "References that compare how a practitioner used to think about themselves as a person or their story that are in contrast with how they currently feel or felt after a shift associated with meditation practice. Also includes references to a complete loss of narrative identity. The narrative self is temporally extended in reference to past or future conceptions of self or personhood. Thus, changes in narrative self include the impact of different perspectives cultivated through the theory and practice of meditation on the stories practitioners tell about themselves. Such instances may benefit from double-coding with Change in Worldview. Stories may include descriptions of changes in how they view the motivations for their behaviors, the type of person that they think they are, or the type of person that they think they should be, in which cases double-coding with Change in Motivation or Goal may also be appropriate.",
            'exclusive': "References to stories of self that do not appear to be emergent from meditation or changed by the theory and practice of meditation should not be coded. Statements in which a practitioner describes himself or herself only in the present moment, without implicit or explicit suggestions of a shift or change from previous conceptions of themselves. Theoretical discussion of topics related to the inclusion criteria but not explicitly connected to the practitioner's sense of himself or herself as a person or to a 'story of me.' Strictly doctrinal or theoretical conceptions of self or person might be better coded as Change in Worldview or as Influencing Factors."
        },
        {
            'category': 'Change in Sense of Embodiment',
            'description': "Feeling of being disembodied, located outside or at a distance from one’s body, or located in an unusual location within one's body schema.",
            'codes': ['Disembodied feeling', 'Altered self-location'],
            'inclusive': "Includes any change in where one's sense of self is located relative to the body schema. Includes references to a change in perspective of location within the body schema, such as from behind the eyes, to the middle of the head or to the heart. May also refer to a distance from the body schema, where the sense of self or perspective is located behind the body or in space, as in 'out-of-body' experiences.",
            'exclusive': "Distinct from reports of expansiveness associated with Change in Self-Other or Self-World Boundaries, as disembodiment still has a particular vantage point or location, and is not everywhere/nowhere. Changes or distortions in body image that do not reflect a change in locatedness of the sense of self would be coded in Somatosensory Changes."
        },
        {
            'category': 'Loss of Sense of Agency',
            'description': "A loss of a sense of ownership or sense of control over one's actions.",
            'codes': ['Lack of control', 'Loss of agency'],
            'inclusive': "Mention of the lack of a 'doer,' or a sense that there is 'no one' in the body who decides, controls, or executes actions, whether habitual  or intentional. References to actions happening on their own or to feeling like a 'puppet,' an automaton, or subject to (the will of) an external force. Also includes references to disconnect from and disidentification with embodied actions such as movement, or to a sense of a being a detached witness/observer to action, but in these cases must also be suggestive of a diminished sense of or loss of agency.",
            'exclusive': "Loss of Sense of Ownership not pertaining to actions should be coded in that category instead. Also to be distinguished from an expanded sense of self in Change in Self-Other or Self-World Boundaries. Care should be taken to disambiguate from or double-code with Meta-Cognition, through which similar changes can occur in a more transient, and less enduring manner, especially in relation to habitual actions. Discussions of energy running through the body and resulting in bodily movements should be coded as or double-coded with Somatic Energy and Involuntary Movements."
        },
        {
            'category': 'Loss of Sense of Ownership',
            'description': "A loss of the usual sense of owning one's thoughts, body sensations, emotions, and/or memories.",
            'codes': ['Loss of ownership', 'Impersonal experiences', 'Anomalous recall'],
            'inclusive': 'Reports that thoughts, body sensations, and/or emotions dont feel like "mine" or like they "belong to me." This may include the feeling that they belong to someone else, or they may be experienced as "impersonal." May include seeing “thoughts as just thoughts” or emotions as impersonal events in the mind or body. Thoughts may no longer become personally relevant and may lose their emotional charge or salience, in which case double-coding with Emotional Detachment may be warranted. Includes the experience of anomalous subjective recall--the experiencing of personal events such that they feel impersonal, or as if they happened to someone else, or to oneself but feel unusually distant in time.',
            'exclusive': 'Some mild forms of "seeing thoughts as thoughts" may be better coded (or double-coded) as Meta-Cognition. Similarly, some forms of distancing from thoughts and the content of thoughts in which a cognitive event is no longer processed with associated meaning or significance may be better coded (or double-coded) as Disintegration of Conceptual Meaning Structures.'
        },
        {
            'category': 'Loss of Sense of Basic Self',
            'description': 'A loss of the sense of existing, of being a self, or of having a self.',
            'codes': ['Loss of basic self', 'Sense of absence', 'Altered unity'],
            'inclusive': 'References to "not being there," to "disappearing," to "being absent," or to "not being present." These changes in sense of self are reported as happening at a very basic level of the sense of self. They may be associated with other types of changes in sense of self, whether concerning embodiment, ownership, or agency, but references should signal an alteration at this basic level of existence or being. Includes references to changes in or problems with the "unity" of various sensory, affective, and cognitive dimensions of experience.',
            'exclusive': 'Changes in sense of self happening at other levels should be coded accordingly. If a loss of ownership or agency is specified, code Loss of Sense of Ownership or Agency. If associated with a change in embodiment is referenced, code Change in Sense of Embodiment or Change in Self-Other or Self-World Boundaries. Unspecified references to changes in sense of self that do not seem to be altering the core sense of existence or being should be coded as Other.'
        },
    #    ]
    #social_domain = [
        {
            'category': 'Change in Relationship to Meditation Community',
            'description': 'Changes in relationship with the meditation community (Sangha), whether increasing or decreasing degrees of affiliation with the community of teacher(s) and other practitioners.',
            'codes': ['Change in affiliation', 'Increased trust in community', 'Decreased trust in community'],
            'inclusive': 'Can include an increased sense of affiliation, belonging or commitment, or an increased sense of trust or faith in the community or teacher, in which cases double-coding with Change in Doubt, Faith, Trust, or Commitment may be warranted. Can also include a change in role towards increasing participation such as becoming a monk or teacher or taking on other institutional roles. Conversely, can also include a decreased sense of affiliation, belonging or commitment, a decreased sense of trust or faith in the community or teacher.',
            'exclusive': 'Social isolation or impairments that are not related to the meditation community, or Sangha. Increased or decreased relationships to other communities would be coded under Increased Sociality or Social Impairment, respectively.'
        },
        {
            'category': 'Increased Sociality',
            'description': 'Increased extraversion, social contact, friendships or other behavioral manifestations indicating an increased valuing of social engagement.',
            'codes': ['Increased extraversion', 'Increased social contact', 'New social activities'],
            'inclusive': 'Clear behavioral manifestations of an increased valuing of social engagement. Examples include increased extraversion, increased social contact or friendships, new social activities, or resuming social activities previously neglected.',
            'exclusive': 'Does not include rhetoric from Buddhist traditions about prosocial values such as compassion or the bodhisattva vow unless accompanied by behavioral changes. Increased sociality not attributed to meditation practice would also not be coded.'
        },
        {
            'category': 'Integration Following Retreat or Intensive Practice',
            'description': 'A destabilizing transition from intensive formal practice to informal practice, daily life, or life circumstances.',
            'codes': ['Transition from practice', 'Re-engagement challenges', 'Desirable retreat experiences'],
            'inclusive': 'Includes any destabilizing transition from formal practice, daily life, or life circumstances, typically following retreat but also including the transition from any period of formal practice back into daily life.',
            'exclusive': 'Any difficulty that arises within a retreat or within daily practice that doesn\'t specifically have to do with re-engagement with one\'s life duties or with actions in daily life.'
        },
        {
            'category': 'Occupational Impairment',
            'description': 'An impaired ability to perform in an occupational environment.',
            'codes': ['Decreased work function', 'Occupational role impairment'],
            'inclusive': 'Describing a decreased ability to function in a normal work environment or fulfill the roles of a job.',
            'exclusive': 'Social impairment or personal feelings of isolation or loneliness that are unrelated to impairment in occupational functioning. If social impairment is directly causing occupational impairment, double-code with Social Impairment.'
        },
        {
            'category': 'Social Impairment',
            'description': 'Behaviors indicative of a change in relationship to social networks or social situations that inhibits ordinary or desired functioning or level of engagement.',
            'codes': ['Decreased social engagement', 'Loss of friendships', 'Decreased social interaction'],
            'inclusive': 'Behavioral manifestations of a subjective feeling of disconnection from social networks (including friends, family, institutions, cultures, religious groups, etc.), or of a general sense of loneliness within said networks, possibly stemming from religious, cultural, emotional, or intellectual differences, or from novel experiences.',
            'exclusive': 'References to social isolation that are voluntary and desired, such as periods of retreat or disassociating from unhealthy relationships, would not be coded here. Describing physical isolation but not including subjective feelings of disconnection from social networks.'
        },
    #]
    #somatic_domain = [
        {
            'category': 'Appetitive or Weight Changes',
            'description': 'Decreased or increased appetite, weight loss or gain.',
            'codes': ['Decreased appetite', 'Increased appetite', 'Weight loss', 'Weight gain'],
            'inclusive': 'Decreased or increased appetite, weight loss or gain attributed to meditation practice. Includes a general disgust with or increased attraction to food. Decreased appetite may be double-coded with Anhedonia or Depression when these co-occur.',
            'exclusive': 'Decreased or increased appetite, weight loss or gain attributed to a cause unrelated to meditation.'
        },
        {
            'category': 'Breathing Changes',
            'description': 'Altered respiration rates that may manifest as a temporary cessation, or speeding up or slowing down of breathing.',
            'codes': ['Respiration changes', 'Breathing irregularity', 'Labored breathing', 'Suffocating feeling'],
            'inclusive': 'Breathing irregularity, either during formal meditation or not, that is attributed to meditation practice. Pre-existing breathing irregularity made worse by meditation practice. Can also include symptoms like labored breathing or a feeling of suffocating. Breathing irregularity may be valenced as positive (calming, relaxing, feeling that it is easier to breathe) or negatively valenced (distressing, feeling of suffocating, etc.)',
            'exclusive': 'Breathing irregularity not attributed to meditation practice. Pre-existing breathing irregularity not made worse by meditation practice. Intentionally controlling or regulating the breathing should not be coded.'
        },
        {
            'category': 'Cardiac Changes',
            'description': 'Irregular heartbeat, heart palpitations, or other significant irregularities.',
            'codes': ['Irregular heartbeat', 'Heart palpitations', 'Tachycardia', 'Brachycardia'],
            'inclusive': 'Cardiac irregularity attributed to meditation practice or pre-existing cardiac irregularity made worse by meditation practice. Cardiac changes include tachycardia (unusually rapid heart beat, even at rest), brachycardia (unusually slow heart beat), and heart palpitations.',
            'exclusive': 'Cardiac irregularity not attributed to meditation practice; pre-existing cardiac irregularity not made worse by meditation practice.'
        },
        {
            'category': 'Dizziness or Syncope',
            'description': 'Dizziness, vertigo (feeling one is spinning or off-balance), lightheadedness (feeling one is about to faint), or syncope (a brief loss of consciousness and muscle strength, commonly called fainting, passing out or blacking out).',
            'codes': ['Dizziness', 'Vertigo', 'Lightheadedness', 'Syncope'],
            'inclusive': 'Dizziness, vertigo, lightheadedness, or syncope attributed to meditation practice or associated with other meditation-related symptoms such as cardiac or breathing irregularities.',
            'exclusive': 'Dizziness, vertigo, lightheadedness, or syncope not attributed to meditation practice. Nausea not resulting from dizziness, lightheadedness or vertigo might be more appropriately coded under or double-coded with Gastrointestinal Distress or Nausea.'
        },
        {
            'category': 'Fatigue or Weakness',
            'description': 'A feeling of exhaustion, fatigue or weakness (general or localized).',
            'codes': ['Exhaustion', 'Fatigue', 'Weakness', 'Chronic fatigue'],
            'inclusive': 'Feelings of exhaustion, fatigue or weakness (general or localized) attributed to meditation practice. Also includes (but not limited to) statements about chronic fatigue, which may co-occur with increased sleep need (and therefore should be double-coded with Sleep Changes). Fatigue may also correlate with cognitive impairments such as Change in Executive Functioning.',
            'exclusive': 'Pre-existing conditions that involve fatigue, or the development of fatigue that is attributed to a source other than meditation (such as Lyme Disease or other medical history).'
        },
        {
            'category': 'Gastrointestinal Distress or Nausea',
            'description': 'Gastrointestinal problems including (but not limited to) diarrhea, bloating, cramping, nausea and vomiting.',
            'codes': ['Diarrhea', 'Bloating', 'Cramping', 'Nausea', 'Vomiting'],
            'inclusive': 'GI distress or nausea attributed to meditation practice or viewed as a consequence of other physiological effects caused by meditation.',
            'exclusive': 'GI distress attributed to diet, location, or other non-meditation causes (e.g., GI issues related to food while on retreat in India). Nausea associated with dizziness, lightheadedness, or vertigo should be double-coded with Dizziness or Syncope.'
        },
        {
            'category': 'Headaches or Head Pressure',
            'description': 'Ache, sharp pain, or pressure in the region of the head or neck.',
            'codes': ['Headaches', 'Head pressure', 'Migraines'],
            'inclusive': 'Headaches or head pressure attributed to or exacerbated by meditation. Headaches include migraines or head pressure often associated with breathing. Includes any brief, prolonged, or intermittent sensations of pressure in the head as well as any associated or subsequent pain or discomfort.',
            'exclusive': 'Headaches or head pressure not attributed to or exacerbated by meditation practice, such as from head trauma or a sinus headache. Pain and Pressure or Tension elsewhere in the body should be coded accordingly with the general categories.'
        },
        {
            'category': 'Involuntary Movements',
            'description': 'A motor movement usually under voluntary control that occurs without a conscious decision for movement.',
            'codes': ['Tics', 'Spasms', 'Twitching', 'Rocking', 'Shaking'],
            'inclusive': 'Involuntary movements that are attributed to meditation practice. This includes spontaneous movements such as: tics, spasms, twitching, rocking, shaking, seizing up, twisting of the torso or head, fidgeting, or others. Includes involuntary vocalizations. May be repetitive or a single involuntary movement. When occurring with reports of Somatic Energy, double-code or code sequentially.',
            'exclusive': 'Involuntary movements not attributed to meditation practice. Differentiate from Loss of Sense of Agency, which is a change in the sense of feeling of ownership over typical or ordinary actions; double-code in cases where it is unclear if the action was considered typical or sporadic, or where involuntary movements subsequently impacted feelings related to sense of self.'
        },
        {
            'category': 'Pain',
            'description': 'Pain is an unpleasant physical sensation, either diffuse or acute, and lasting for variable amounts of time.',
            'codes': ['Diffuse pain', 'Acute pain', 'Postural aches'],
            'inclusive': 'Pain of any kind attributed to meditation practice beyond typical and expected postural aches and pain. Include also the relieving of pain attributed to meditation. When pain or the alleviation of pain is mentioned in conjunction with Pressure, Tension or Release of Pressure, Tension double-code these two together.',
            'exclusive': 'Pain not attributed to meditation practice. Expected or anticipated levels of postural pain arising on account of the somatic immobility entailed in the practice of meditation and that is casually mentioned without attribution of significance or downstream consequences. Pain in the head and neck area should be coded under Headaches.'
        },
        {
            'category': 'Parasomnias',
            'description': 'Nightmares, vivid dreams, sleep paralysis or the alleviation of these symptoms.',
            'codes': ['Nightmares', 'Vivid dreams', 'Sleep paralysis'],
            'inclusive': 'Includes nightmares, vivid dreams or sleep paralysis (or experiences that resemble sleep paralysis), or the alleviation of these symptoms, attributed to meditation practice. Nightmares or vivid dreams includes either unpleasant dreams that can cause a strong negative emotional response or dreams with unusually good recall that are often perceived as if a recent experience. Sleep paralysis occurs as one falls asleep or as one wakes: an open-eyed paralysis that can last from seconds to minutes and is often accompanied by a sense of terror, danger and/or hallucinations.',
            'exclusive': 'Nightmares, vivid dreams or sleep paralysis or the alleviation of these symptoms attributed to a cause unrelated to meditation--for instance, cases of ongoing sleep paralysis unaffected by meditation practice. Changes in sleep amount, need, or insomnia should be coded under Sleep Changes.'
        },
        {
            'category': 'Pressure, Tension or Release of Pressure, Tension',
            'description': 'Bodily pressure or tension, or release of bodily pressure or tension, that can vary according to location (general or specific), intensity, or length of time.',
            'codes': ['Pressure', 'Tension', 'Release of pressure', 'Release of tension'],
            'inclusive': 'Pressure or tension, or release of pressure or tension, attributed to meditation practice. Pressure or tension includes pressure "knots," which are sensations of intense energy in acute body locations that feel as though there are energy blockages, and may include a sensation of twisting. It also includes experiences of tightness, constriction, squeezing, contraction, locking of muscles, and other manifestations of pressure or tension. Release of pressure or tension may be described as “opening,” “relaxation” or “release,” and is sometimes but not necessarily linked to Re-experiencing of Traumatic Memories and may be double-coded there or with Somatic Energy or Involuntary Movements.',
            'exclusive': 'Pressure or tension, or release of pressure or tension, not attributed to meditation practice. Head Pressure connected to breathing is coded in its own category. Ordinary states of physical relaxation not characterized as a release of pressure or tension in specific areas of the body.'
        },
        {
            'category': 'Sexuality-Related Changes',
            'description': 'Hypersexuality (very frequent or suddenly increased sexual urges or activity) or hyposexuality (notably decreased sexual urges or activity).',
            'codes': ['Hypersexuality', 'Hyposexuality', 'Unwanted sexual thoughts'],
            'inclusive': 'Sexuality-related changes attributed to meditation practice. Hypersexuality can also include increased intensity of sexual urges or activity, a sense of being sexually out of control (or that one is not controlling sexual activity oneself), unwanted or disturbing thoughts that are sexual in content, or otherwise notably unusual changes related to increases in sexual urges or activity. Subjects may connect hypersexuality to "tantric" insights, to "kundalini" awakenings, or other Somatic Energy experiences and effects. Thus, double-code with Somatic Energy where appropriate.',
            'exclusive': 'Sexuality-related changes not attributed to meditation practice. Descriptions of sexual energy not leading to a change in sexual desire or sexual behavior might be more appropriately coded as Somatic Energy.'
        },
        {
            'category': 'Sleep Changes',
            'description': 'Changes in sleep amount, sleep need, or sleep depth.',
            'codes': ['Insomnia', 'Decreased sleep need', 'Hypersomnia'],
            'inclusive': 'Changes in sleep amount or depth (including intensification of existing symptoms) attributed to meditation practice. Includes (but not limited to): difficulty falling asleep or staying asleep through the night, insomnia, decreased or increased sleep need, hypersomnia (excessive daytime sleepiness or prolonged nighttime sleep), or other related symptoms. May be double-coded with Fatigue or with Parasomnias.',
            'exclusive': 'Changes in sleep amount or depth not attributed to meditation practice. Changes in dreams, nightmares, and other anomalous sleep experiences would be coded under Parasomnias.'
        },
        {
            'category': 'Somatic Energy',
            'description': 'A type of sensation moving throughout the body or throughout a body area described with language of vibration, energy, current, or other related metaphors.',
            'codes': ['Somatic energy', 'Vibrations', 'Currents'],
            'inclusive': 'Somatic energy or vibrations attributed to meditation practice. Include any level of intensity (gentle to forceful, flowing to surging.) May be described as being contained within or going beyond one’s body schema. May also be reported as under one’s control or involuntary, in which case double-coding with Loss of Sense of Agency may be appropriate. May include use of emic terms like "kundalini," "prana," or metaphors of "currents," "voltage," or "electricity." Also includes related references to psychomotor agitation (adrenaline rushes, feeling "wired") or slowing. May also be double-coded with Release of Tension or with Agitation. Reports of somatic energy may also affect motor patterns and may therefore be closely associated with reports of Involuntary Movements.',
            'exclusive': 'Somatic energy or vibrations that are not attributed to meditation practice. The use of "energy" as a metaphysical concept without any associated phenomenology should not be coded. The term "vibrations" being used to describe minor twitches and other minor somatic movements that may also or alternatively be classified under Involuntary Movements.'
        },
        {
            'category': 'Thermal Changes',
            'description': 'Changes associated with heat or cold, whether a general change in sense of body temperature or localized to a specific body area.',
            'codes': ['Sweating', 'Overheating', 'Cold sweats', 'Goosebumps'],
            'inclusive': 'Thermal changes attributed to meditation practice. Includes (but not limited to) statements about: sweating, dry mouth, overheating, hot flashes, prickling sensations, burning sensations, cold sweats, goosebumps, chills, hair standing on end, feeling unusually cold, or other related symptoms. Classic "kundalini" experiences may have heat or cold accompanied by Somatic Energy.',
            'exclusive': 'Thermal changes not attributed to meditation practice, such as changes in environmental temperature.'
        }
    ]

    #phenemonology_codebook = affective_domain, cognitive_domain + perceptual_domain + social_domain + sense_of_self_domain + somatic_domain

    study_codebook = [
        
            {
                "category": "Novel Sensation",
                "description" : "",
                "codes": ["The experience of new or unfamiliar sensations, particularly in relation to pain or the body."],
                "inclusive": "new, unfamiliar, or unexpected sensations experienced in the body or in relation to pain. This could include changes in the quality, intensity, or location of pain, or new bodily sensations associated with mindfulness practices. ",
                "exclusive": "external novelty not related to mindfulness practices or the individual's own bodily experience (e.g., new tastes or smells, external sensory experiences).",
            },
            {
                "category": "Disability",
                "description" : "The state of being unable to do something, particularly in relation to managing pain or engaging in mindfulness practices.",
                "codes": ["The presence of a physical or mental condition that limits a persons movements, senses, or activities.", "disabling pain, hurt"],
                "inclusive": "physical or mental conditions that limit the individuals activities, particularly in relation to pain or mindfulness practices. This could include limitations in mobility, sensory perception, cognitive function, or other aspects of functioning. ",
                "exclusive": "disability that are not related to the individual's own physical or mental condition (e.g., societal or environmental barriers)."
            },
            {
                "category": "Self-Efficacy",
                "description" : "",
                "codes": ["The belief in one's ability to succeed in specific situations or accomplish a task. "],
                "inclusive": "beliefs about one's own ability to manage pain, engage in mindfulness practices, or cope with disability. This could include confidence in one's ability to control pain, adhere to mindfulness practices, or overcome challenges related to disability. ",
                "exclusive": "self-efficacy not related to pain, mindfulness practices, or disability (e.g., confidence in unrelated skills or abilities)."
            },
            {
                "category": "Barriers",
                "description" : "",
                "codes": ["Obstacles ", " impediments that hinder or prevent engagement in mindfulness practices. "],
                "inclusive": "factors that make it difficult to engage in mindfulness practices, such as physical pain, time constraints, noisy environments, or technical issues during sessions. This could also include psychological barriers, such as fear or negative beliefs about the practice.",
                "exclusive": "barriers that are not related to mindfulness practices (e.g., barriers to unrelated activities or tasks)."
            },
            {
                "category": "Distractions",
                "description" : "",           
                "codes": ["Factors that divert attention away from mindfulness practices. ", "The presence of unwanted or disruptive noise during mindfulness practices.", "Challenges making time"],
                "inclusive": "external or internal factors that distract from mindfulness practices, such as noise, interruptions, physical discomfort, or intrusive thoughts. This could also include distractions during meditation sessions, such as technical issues or participant oversharing. References to noise or sound that disrupts mindfulness practices, such as background noise, interruptions from others, or technical issues during sessions. This could also include the challenge of practicing mindfulness in a noisy or busy environment. ",
                "exclusive": "noise that is not related to mindfulness practices (e.g., noise during unrelated activities or tasks). Any distractions that are not related to mindfulness practices (e.g., distractions during unrelated activities or tasks)."
            },
            {
                "category": "Fear",
                "description" : "",
                "codes": ["Perception of potential pain, discomfort. in this context refers to the anticipation or perception of potential pain or discomfort that may arise from mindfulness practices or the condition of chronic pain itself."],
                "inclusive": " apprehension, anxiety, or fear related to their pain, the potential for increased pain, or discomfort associated with mindfulness practices. This could be fear of the unknown, fear of potential pain exacerbation, or fear of not being able to manage or cope with their pain.",
                "exclusive": "general fears or anxieties unrelated to their chronic pain or mindfulness practices. Fear that is not directly linked to the perception of potential pain or discomfort, such as fear of social situations or other unrelated anxieties, should not be included."
            },
            {
                "category": "Fear Avoidance",
                "description" : "",
                "codes": ["suffering", "attachment", "helplessness", "refers to behaviors that individuals engage in to avoid situations or activities that they fear may cause them pain or increase their suffering. This can include physical activities that they believe may exacerbate their pain, or mental/emotional situations that they believe may increase their suffering.", "The tendency to avoid pain or activities that are believed to cause pain due to fear or anxiety. "],
                "inclusive": " avoidance of certain activities or situations due to fear of increased pain or suffering. This could be avoidance of physical activities, avoidance of certain mindfulness practices, or avoidance of mental/emotional situations that they believe may increase their suffering. avoidance of pain or pain-related activities due to fear, anxiety, or negative beliefs about pain. This could include avoidance of physical activities, social situations, or other aspects of life due to fear of a potential pain.",
                "exclusive": "avoidance behaviors that are not driven by fear of increased pain or suffering. For example, avoidance of activities due to lack of interest or motivation should not be included, fear-avoidance that are not related to pain or the individual's own beliefs or emotions (e.g., fear-avoidance of non-pain-related activities or situations)."
            },
            {
                "category": "Pain Avoidance",
                "description" : "",
                "codes": [" self-efficacy","non-violence"," right action.", " refers to the behaviors and strategies that individuals living with chronic pain engage in to avoid experiencing pain. This can include both physical strategies (such as avoiding certain movements or activities) and psychological strategies (such as cognitive reappraisal or distraction techniques)."],
                "inclusive": " avoidance of certain activities or situations due to fear of pain, or uses cognitive strategies to manage their pain. This could be avoidance of physical activities, use of distraction techniques, or cognitive reappraisal to manage their perception of pain.",
                "exclusive": "avoidance behaviors that are not driven by fear of pain. For example, avoidance of activities due to lack of interest or motivation should not be included. It also does not include pain management strategies that do not involve avoidance, such as acceptance or mindfulness techniques."
            },
            {
                "category": "Reappraisal",
                "description" : "",
                "codes": ["acceptance", "non-attachment", "equanimity", "This refers to the cognitive process of reinterpreting a situation or experience in order to change its emotional impact. In the context of chronic pain and mindfulness, this could involve changing ones perspective on their pain, accepting it rather than resisting it, or cultivating a state of equanimity towards it."],
                "inclusive": " a change in  perspective on  pain, a shift towards acceptance or non-attachment, or the cultivation of equanimity. This could be a shift from viewing pain as a threat to viewing it as a neutral sensation, or a shift from resisting pain to accepting it.",
                "exclusive": "cognitive strategies that do not involve reappraisal, such as distraction or avoidance. It also does not include reappraisal strategies that are not related to pain or mindfulness, such as reappraising a social situation or a non-pain-related physical sensation."
            }
        ]

    test_codebook = [
            {
                'category': 'Reappraisal',
                "description" : "Acceptance of the situation, exactly as it is, with all the good and bad.",
                'codes': ['reappraisal', 'non-attachment', 'equanimity'],
                'inclusive': ' a change in the perspective on pain, a shift towards acceptance or non-attachment, or the cultivation of equanimity. This could be a shift from viewing pain as a threat to viewing it as a neutral sensation, or a shift from resisting pain to accepting it.',
                'exclusive': 'distraction or avoidance. It also does not include reappraisal strategies that are not related to pain or mindfulness, such as reappraising a social situation or a non-pain-related physical sensation.'
            },
            {
                'category': 'Pain Avoidance',
                "description" : "avoidance for the sake of pain that could be caused if action is taken",
                'codes': ['self-efficacy', 'non-violence', 'right action'],
                'inclusive': 'avoidance of activity due to fear of pain, or uses cognitive strategies to manage pain. This could be avoidance of physical activities, use of distraction techniques, or cognitive reappraisal to manage their perception of pain.',
                'exclusive': 'avoidance behaviors that are not driven by fear of pain. For example, avoidance of activities due to lack of interest or motivation should not be included. It also does not include pain management strategies that do not involve avoidance, such as acceptance or mindfulness techniques.'
            },
            {
                'category': 'Fear Avoidance',
                "description" : "avoidance for the sake of fear that could be caused if action is taken",
                'codes': ['suffering', 'attachment', 'helplessness'],
                'inclusive': 'avoidance of activity due to fear of increased pain or suffering. This could be avoidance of physical activities, avoidance of certain mindfulness practices, or avoidance of mental/emotional situations that they believe may increase their suffering.',
                'exclusive': 'avoidance behaviors that are not driven by fear of increased pain or suffering. For example, avoidance of activities due to lack of interest or motivation should not be included.'
            }
        ]
    if which is None:
        return ([test_codebook, study_codebook, phenomenology_codebook])
    elif "full" and "study" in which:
        return [study_codebook, phenomenology_codebook]
    elif "study" and "test" in which:
        return [study_codebook, test_codebook]
    elif which == "test":
        return [test_codebook]
    elif which == "study":
        return [study_codebook]
    elif which == "full":
        return [phenomenology_codebook]
    
    
import argparse


if __name__ == "__main__":
    output="ALGORITHMICALLY_APPLIED_RESEARCH_CODES_"

    
    errorsrts=" 3.srt 27.srt 30.srt"
    instructor_only_file_uncoded=" 9.srt 10.srt 15.srt "
    srt_files=["1.srt","2.srt","4.srt","5.srt","6.srt","7.srt","8.srt", "12.srt", "13.srt", "14.srt", "16.srt", "17.srt", "18.srt", "19.srt", "20.srt", "21.srt", "22.srt", "23.srt", "24.srt", "25.srt", "26.srt", "28.srt", "29.srt"]
    codebooks_to_try = get_codebook("study")
    coded_output_only=False
    concat_and_condense=False
    distance_threshold=1.325
    similarity_threshold=1.375
    sentiment_threshold=1.35

    if concat_and_condense:
        concatname="_".join("".join(srt_files).split(".srt"))
        srt_files=[concat_srt_files(srt_files, "srt_files_include_"+concatname+".srt")]
        coded_output_only=True
 

    veto_stats=(0,0,0,0,0,0)
    out = []
    for i, codes in enumerate(codebooks_to_try):
        coded_text_list=[]
        results=f"{output}{i}.txt"
        count_codes_applied = []
        for i, srt_file in enumerate(srt_files):
    #        try:
                print("SRT", srt_file)
                srtf=(srt_file)
                coded_sents, veto_stats, codes_applied_list=(apply_research_codes_to_sentences(srt_file = srt_file, codes=codes, coded_output_only=coded_output_only, veto_stats=veto_stats, bias_critera_exclusion_weight=.1, max_codes_per_sentence=9, sentiment_threshold=sentiment_threshold ,    distance_threshold=distance_threshold, similarity_threshold=similarity_threshold
    ))
                for code_applied in codes_applied_list:
                    count_codes_applied.append(code_applied)
                coded_text_list.append("\n\n\n_____\n"+srtf+"\n\n")
                count_dict = Counter(codes_applied_list)
                
                count = ( ('\n'.join(f"{key}: {value}\n" for key, value in sorted(list(count_dict.items()),  key=lambda item: item[1],reverse=True))))
                coded_text_list.append(count)
                coded_text_list.append("\n"+coded_sents)
        #       except:
        #          print("FAILED SRT", srt_file, "ITER", i)
        cdict = Counter(count_codes_applied)
        #count_codes_applied = list(count_dict.items())
        all_results = ( str('\n'.join(f"{value}: {key}" for key, value in sorted(list(cdict.items()), key=lambda item: item[1], reverse=True))))


        vd, vsen, vsim, total_codecount,total_sub_list_len, total_sents = veto_stats
        print("number_vetoed_by_distance",vd)
        print("number_vetoed_by_sentiment",vsen)
        print("number_vetoed_by_similarity",vsim)
    #  print("total_codecount",total_codecount)
    #   print("total total_sub_list_len",total_sub_list_len)
    #    print("total_sents",total_sents)

        print("\n\nAvg num labels applied each subtitle which recieved one label:", total_codecount/total_sub_list_len) if total_sub_list_len !=0 else print("ERROR no subs")
        print("PERCENTAGE OF SENTS CODED:", total_sub_list_len/total_sents) if total_sents !=0 else print("ERROR no sents")
        print("research code occurrences:\n",all_results)
        with open(results, "w+") as f:
            if len(srt_files)>1:
                f.write("\n\n SRT_TRANSCRIPTS_CODED SUCCESSFULLY:" + str("".join(srt_files)))
                f.write("\n__________\n\n ALL_RESULTS:\n" + all_results + "\n__________\n\n")
            if total_sents!=0:
                f.write(f"Percentage of sentences which were coded:{total_sub_list_len/total_sents}")
            if total_sub_list_len!=0:
                f.write(f"\n\nAvg num codes applied to each coded sub : {total_codecount/total_sub_list_len}")

                f.write("\nResults per transcript:\n")
            
            for coded_text in coded_text_list:
                    f.write(coded_text)
            f.write("\n\n\n\n\n_____________________________________")
            f.write(f"\n NOTE: transcripts with instructor only have been skipped: {instructor_only_file_uncoded}\n\n\n")
            f.write(f"\n WARNING: Unmatched SRT formatting\n skipping these files: {errorsrts}\n\n\n")
            f.write(" the embeddings are generated using open source 'bert-base-uncase',sentiment_tokenizer = cardiffnlp/twitter-roberta-base-sentiment) sentiment_model = AutoModelForSequenceClassification.from_pretrained(cardiffnlp/twitter-roberta-base-sentiment)")
            f.write("Embeddings calculation is done with a struct of category:definition|codes|+include:-exclude: \n This assigns a weight to the concatinated vector embeddings of category:definition|codes|, with a half-weight to the +inclusive: criteria embeddings and a negative half-weight to the -exclusive: criteria embeddings.")
            f.write(f"\n\nSTATS:\ncount of proposed_codes_vetoed_by_distance{ vd }, as based on the sensitivity parameter for distance threshold, manually set with a weight of {distance_threshold}, which is multiplied by the greater of (the average distance of that code to any sentence/subtitle, or, the average of all codes to all subtitles) to determine the threshold for vetoing a proposed best similarity match or sentiment match to code.")
            f.write(f"\ncount of proposed_codes_vetoed_by_sentiment {vsen}, as is based on the sensitivity parameter for sentiment_likeness with a weight of {sentiment_threshold} (#note the exclusive prompt is missing from sentiment). The threshold for sentiment is multiplied by the greater of (the average sentiment difference between all subtitles and the proposed research code, or, the average sentiment difference of all codes against all subtitles). If the code and sentence are less sentimentally alike than the greatest average, then the sentiment threshold is not met and the code is not applied.  ")
            f.write(f"\nproposed_codes_vetoed_by_similarity {vsim}, as is based on the similarity sensitivity parameter with a weight of {similarity_threshold}, which is multiplied by the greater average calculation to determine a threshold, under which the similarity of sentences would prevent the code from being applied\n")

            f.write(f"\n\n____________ APPLIED CODEBOOK definitions:\n")
            for i, code in enumerate(codes):
                f.write(f"{code['category']} : ") 
                f.write(f" {code['description']}. \n") 
                f.write(f" | codes |  {','.join(code['codes'])} \n")  
                f.write(f" + include criteria : {str(code['inclusive'])} \n") 
                f.write(f" - exclude criteria : {str(code['exclusive'])} \n")
                if len(codes)>=i+2:
                    f.write(f"_{i+2}\n")

        print(results)
        out.append(results)
    print(out)
