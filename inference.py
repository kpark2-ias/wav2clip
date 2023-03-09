import wav2clip
import numpy as np
import glob
import librosa
from tqdm import tqdm       
from utils.extract_audio import extract_audio
import faiss
import torch
import os
import clip
import pdb
        

class Wav2CLIPInference(object):
    
    def __init__(self, labels, verbose=True):
        self.labels = labels
        self.model = wav2clip.get_model()
        
        if verbose:
            parameters = sum([x.numel()
                              for x in self.model.parameters()]) / (10 ** 6)
            print(f'Parameter count: {parameters:.1f}M')
        
    
    def obtain_embeddings(self, audio, text_features=None, faiss_index=True):

        audio = audio.numpy()
        audio_features = wav2clip.embed_audio(audio, self.model)
        
        if faiss_index == True:
            return audio_features
        else:
            return self.calculate_similarity(audio_features, text_features)
            
        
    def calculate_similarity(self, audio_features, text_features):
        
        text_features = text_features.float()
        
        logits_audio_text = audio_features @ text_features.T
        
        return logits_audio_text

    def preprocess_audio(self,
                         input_dir,
                         SAMPLE_RATE=44100,
                         verbose=False,
                         batch_size=1 << 6,
                         **kwargs):

        paths_to_audio = glob.glob(f'{input_dir}/*.wav')
        audio = list()
        audio_paths = []
        for idx, path_to_audio in enumerate(tqdm(paths_to_audio)):
            
            track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
            track = torch.from_numpy(track)

            audio.append(track)
            audio_paths.append(path_to_audio)

            if not idx % batch_size:
                tracks = audio

                if verbose:
                    print([track.shape for track in tracks])
                    
                maxtrack =  max([ta.shape[-1] for ta in tracks])

                padded = [torch.nn.functional.pad(torch.from_numpy(np.array(ta)),(0,maxtrack-ta.shape[-1])) for ta in tracks]

                if verbose:
                    print( [track.shape for track in padded])

                audio = torch.stack(padded)

                if verbose:
                    print(audio.shape)
                
                yield audio, audio_paths
                    
                audio = []
                audio_paths = []
                
    def score_inputs(self, logits_audio, paths_to_audio, trained_faiss_index=None, faiss_index=True):
        print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)',
              end='\n\n')

        if faiss_index==True:
            k = 1
            distances, indices = trained_faiss_index.search(logits_audio.cpu(), k)

            for audio_idx in range(len(paths_to_audio)):
                #conf_values, ids = confidence[audio_idx].topk(1)

                query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
                results = ', '.join([
                    f'{self.labels[i]:>15s} ({d:06.2%})'
                    for d, i in zip(distances[audio_idx], indices[audio_idx])
                ])

                print(query + results)
        else:
            confidence = logits_audio.softmax(dim=1)
            for audio_idx in range(len(paths_to_audio)):
                conf_values, ids = confidence[audio_idx].topk(1)

                query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
                results = ', '.join([
                    f'{self.labels[i]:>15s} ({v:06.2%})'
                    for v, i in zip(conf_values, ids)
                ])

                print(query + results)
                 
    def __call__(self, input_dir=None, faiss_index=True, verbose=True, **kwargs):
       
        audio_dir = extract_audio(input_dir, **kwargs)
        
        if faiss_index == True:
            trained_faiss_index = faiss.read_index('faiss-index-audioset-527.index')
            for audio, paths_to_audio in self.preprocess_audio(audio_dir,
                                                           verbose=verbose,
                                                           **kwargs):
                logits_audio = self.obtain_embeddings(audio, faiss_index)
                self.score_inputs(logits_audio, paths_to_audio,
                                  trained_faiss_index)
        else:
            device='cuda'
            model, preprocess = clip.load("ViT-B/32", device=device)
            text_inputs = clip.tokenize(labels).to(device)
    
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)

            for audio, paths_to_audio in self.preprocess_audio(audio_dir,
                                                               verbose=verbose,
                                                               **kwargs):
                logits_audio_text = self.obtain_embeddings(audio, text_features)
                self.score_inputs(logits_audio_text, paths_to_audio)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=1 << 6,
                        help='batch size')
    parser.add_argument('-f', '--input_dir')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='increase verbosity')
    parser.add_argument('-n',
                        '--num',
                        type=int,
                        help='Limit to first N input files')
    parser.add_argument('-i', '--faiss_index', 
                        default=True,
                        help='use faiss index optimization')
    args = parser.parse_args()

    labels = ['Music', 'Speech', 'Vehicle', 'Musical instrument', 'Plucked string instrument', 'Singing', 'Car', 'Animal', 'Outside, rural or natural', 'Violin, fiddle', 'Bird', 'Drum', 'Engine', 'Narration, monologue', 'Drum kit', 'Acoustic guitar', 'Dog', 'Child speech, kid speaking', 'Bass drum', 'Rail transport', 'Motor vehicle (road)', 'Water', 'Female speech, woman speaking', 'Siren', 'Railroad car, train wagon', 'Tools', 'Silence', 'Snare drum', 'Wind', 'Bird vocalization, bird call, bird song', 'Fowl', 'Wind instrument, woodwind instrument', 'Emergency vehicle', 'Laughter', 'Chirp, tweet', 'Rapping', 'Cheering', 'Gunshot, gunfire', 'Radio', 'Cat', 'Hi-hat', 'Helicopter', 'Fireworks', 'Stream', 'Bark', 'Baby cry, infant cry', 'Snoring', 'Train horn', 'Double bass', 'Explosion', 'Crowing, cock-a-doodle-doo', 'Bleat', 'Computer keyboard', 'Civil defense siren', 'Bee, wasp, etc.', 'Bell', 'Chainsaw', 'Oink', 'Tick', 'Tabla', 'Liquid', 'Traffic noise, roadway noise', 'Beep, bleep', 'Frying (food)', 'Whack, thwack', 'Sink (filling or washing)', 'Burping, eructation', 'Fart', 'Sneeze', 'Aircraft engine', 'Arrow', 'Giggle', 'Hiccup', 'Cough', 'Cricket', 'Sawing', 'Tambourine', 'Pump (liquid)', 'Squeak', 'Male speech, man speaking', 'Keyboard (musical)', 'Pigeon, dove', 'Motorboat, speedboat', 'Female singing', 'Brass instrument', 'Motorcycle', 'Choir', 'Race car, auto racing', 'Chicken, rooster', 'Idling', 'Sampler', 'Ukulele', 'Synthesizer', 'Cymbal', 'Spray', 'Accordion', 'Scratching (performance technique)', 'Child singing', 'Cluck', 'Water tap, faucet', 'Applause', 'Toilet flush', 'Whistling', 'Vacuum cleaner', 'Meow', 'Chatter', 'Whoop', 'Sewing machine', 'Bagpipes', 'Subway, metro, underground', 'Walk, footsteps', 'Whispering', 'Crying, sobbing', 'Thunder', 'Didgeridoo', 'Church bell', 'Ringtone', 'Buzzer', 'Splash, splatter', 'Fire alarm', 'Chime', 'Babbling', 'Glass', 'Chewing, mastication', 'Microwave oven', 'Air horn, truck horn', 'Growling', 'Telephone bell ringing', 'Moo', 'Change ringing (campanology)', 'Hands', 'Camera', 'Pour', 'Croak', 'Pant', 'Finger snapping', 'Gargling', 'Inside, small room', 'Outside, urban or manmade', 'Truck', 'Bowed string instrument', 'Medium engine (mid frequency)', 'Marimba, xylophone', 'Aircraft', 'Cello', 'Flute', 'Glockenspiel', 'Power tool', 'Fixed-wing aircraft, airplane', 'Waves, surf', 'Duck', 'Clarinet', 'Goat', 'Honk', 'Skidding', 'Hammond organ', 'Electronic organ', 'Thunderstorm', 'Steelpan', 'Slap, smack', 'Battle cry', 'Percussion', 'Trombone', 'Banjo', 'Mandolin', 'Guitar', 'Strum', 'Boat, Water vehicle', 'Accelerating, revving, vroom', 'Electric guitar', 'Orchestra', 'Wind noise (microphone)', 'Effects unit', 'Livestock, farm animals, working animals', 'Police car (siren)', 'Rain', 'Printer', 'Drum machine', 'Fire engine, fire truck (siren)', 'Insect', 'Skateboard', 'Coo', 'Conversation', 'Typing', 'Harp', 'Thump, thud', 'Mechanisms', 'Canidae, dogs, wolves', 'Chuckle, chortle', 'Rub', 'Boom', 'Hubbub, speech noise, speech babble', 'Telephone', 'Blender', 'Whimper', 'Screaming', 'Wild animals', 'Pig', 'Artillery fire', 'Electric shaver, electric razor', 'Baby laughter', 'Crow', 'Howl', 'Breathing', 'Cattle, bovinae', 'Roaring cats (lions, tigers)', 'Clapping', 'Alarm', 'Chink, clink', 'Ding', 'Toot', 'Clock', 'Children shouting', 'Fill (with liquid)', 'Purr', 'Rumble', 'Boing', 'Breaking', 'Light engine (high frequency)', 'Cash register', 'Bicycle bell', 'Inside, large room or hall', 'Domestic animals, pets', 'Bass guitar', 'Electric piano', 'Trumpet', 'Horse', 'Mallet percussion', 'Organ', 'Bicycle', 'Rain on surface', 'Quack', 'Drill', 'Machine gun', 'Lawn mower', 'Smash, crash', 'Trickle, dribble', 'Frog', 'Writing', 'Steam whistle', 'Groan', 'Hammer', 'Doorbell', 'Shofar', 'Cowbell', 'Wail, moan', 'Bouncing', 'Distortion', 'Vibraphone', 'Air brake', 'Field recording', 'Piano', 'Male singing', 'Bus', 'Wood', 'Tap', 'Ocean', 'Door', 'Vibration', 'Television', 'Harmonica', 'Basketball bounce', 'Clickety-clack', 'Dishes, pots, and pans', 'Crumpling, crinkling', 'Sitar', 'Tire squeal', 'Fly, housefly', 'Sizzle', 'Slosh', 'Engine starting', 'Mechanical fan', 'Stir', 'Children playing', 'Ping', 'Owl', 'Alarm clock', 'Car alarm', 'Telephone dialing, DTMF', 'Sine wave', 'Thunk', 'Coin (dropping)', 'Crunch', 'Zipper (clothing)', 'Mosquito', 'Shuffling cards', 'Pulleys', 'Toothbrush', 'Crowd', 'Saxophone', 'Rowboat, canoe, kayak', 'Steam', 'Ambulance (siren)', 'Goose', 'Crackle', 'Fire', 'Turkey', 'Heart sounds, heartbeat', 'Singing bowl', 'Reverberation', 'Clicking', 'Jet engine', 'Rodents, rats, mice', 'Typewriter', 'Caw', 'Knock', 'Ice cream truck, ice cream van', 'Stomach rumble', 'French horn', 'Roar', 'Theremin', 'Pulse', 'Train', 'Run', 'Vehicle horn, car horn, honking', 'Clip-clop', 'Sheep', 'Whoosh, swoosh, swish', 'Timpani', 'Throbbing', 'Firecracker', 'Belly laugh', 'Train whistle', 'Whistle', 'Whip', 'Gush', 'Biting', 'Scissors', 'Clang', 'Single-lens reflex camera', 'Chorus effect', 'Inside, public space', 'Steel guitar, slide guitar', 'Waterfall', 'Hum', 'Raindrop', 'Propeller, airscrew', 'Filing (rasp)', 'Reversing beeps', 'Shatter', 'Sanding', 'Wheeze', 'Hoot', 'Bow-wow', 'Car passing by', 'Tick-tock', 'Hiss', 'Snicker', 'Whimper (dog)', 'Shout', 'Echo', 'Rattle', 'Sliding door', 'Gobble', 'Plop', 'Yell', 'Drip', 'Neigh, whinny', 'Bellow', 'Keys jangling', 'Ding-dong', 'Buzz', 'Scratch', 'Rattle (instrument)', 'Hair dryer', 'Dial tone', 'Tearing', 'Bang', 'Noise', 'Bird flight, flapping wings', 'Grunt', 'Jackhammer', 'Drawer open or close', 'Whir', 'Tuning fork', 'Squawk', 'Jingle bell', 'Smoke detector, smoke alarm', 'Train wheels squealing', 'Caterwaul', 'Mouse', 'Crack', 'Whale vocalization', 'Squeal', 'Zither', 'Rimshot', 'Drum roll', 'Burst, pop', 'Wood block', 'Harpsichord', 'White noise', 'Bathtub (filling or washing)', 'Snake', 'Environmental noise', 'String section', 'Cacophony', 'Maraca', 'Snort', 'Yodeling', 'Electric toothbrush', 'Cupboard open or close', 'Sound effect', 'Tapping (guitar technique)', 'Ship', 'Sniff', 'Pink noise', 'Tubular bells', 'Gong', 'Flap', 'Throat clearing', 'Sigh', 'Busy signal', 'Zing', 'Sidetone', 'Crushing', 'Yip', 'Gurgling', 'Jingle, tinkle', 'Boiling', 'Mains hum', 'Humming', 'Sonar', 'Gasp', 'Power windows, electric windows', 'Splinter', 'Heart murmur', 'Air conditioning', 'Pizzicato', 'Ratchet, pawl', 'Chirp tone', 'Heavy engine (low frequency)', 'Rustling leaves', 'Speech synthesizer', 'Rustle', 'Clatter', 'Slam', 'Eruption', 'Cap gun', 'Synthetic singing', 'Shuffle', 'Wind chime', 'Chop', 'Scrape', 'Squish', 'Foghorn', "Dental drill, dentist's drill", 'Harmonic', 'Static', 'Sailboat, sailing ship', 'Cutlery, silverware', 'Gears', 'Chopping (food)', 'Creak', 'Fusillade', 'Roll', 'Electronic tuner', 'Patter', 'Electronic music', 'Dubstep', 'Techno', 'Rock and roll', 'Pop music', 'Rock music', 'Hip hop music', 'Classical music', 'Soundtrack music', 'House music', 'Heavy metal', 'Exciting music', 'Country', 'Electronica', 'Rhythm and blues', 'Background music', 'Dance music', 'Jazz', 'Mantra', 'Blues', 'Trance music', 'Electronic dance music', 'Theme music', 'Gospel music', 'Music of Latin America', 'Disco', 'Tender music', 'Punk rock', 'Funk', 'Music of Asia', 'Drum and bass', 'Vocal music', 'Progressive rock', 'Music for children', 'Video game music', 'Lullaby', 'Reggae', 'New-age music', 'Christian music', 'Independent music', 'Soul music', 'Music of Africa', 'Ambient music', 'Bluegrass', 'Afrobeat', 'Salsa music', 'Music of Bollywood', 'Beatboxing', 'Flamenco', 'Psychedelic rock', 'Opera', 'Folk music', 'Christmas music', 'Middle Eastern music', 'Grunge', 'Song', 'A capella', 'Sad music', 'Traditional music', 'Scary music', 'Ska', 'Chant', 'Carnatic music', 'Swing music', 'Happy music', 'Jingle (music)', 'Funny music', 'Angry music', 'Wedding music', 'Engine knocking']
    self = Wav2CLIPInference(labels)
    self(**vars(args))
