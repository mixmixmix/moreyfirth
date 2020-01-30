#Main file for analysis
import salmomiks
import glob

filtered_file = 'data/filtered.csv'
fish_file = 'data/fish.csv'

salmomiks.load_data('data/recievers.csv'
          , glob.glob('data/observations/*csv')
          , filtered_file
)

salmomiks.make_rec_map(filtered_file, 'data/recmap.html')

salmomiks.make_fish_df(filtered_file, fish_file)
