#Main file for analysis
import salmomiks
import glob

filtered_file = 'data/filtered.csv'
fish_file = 'data/fish.csv'
tag_file = 'data/moray_tag_info.csv'
fish_file = 'data/fish_moove.csv'

#salmomiks.load_data('data/recievers.csv'
#                    , tag_file
#                    , glob.glob('data/observations/*csv')
#                    , filtered_file
#)
#
#salmomiks.make_rec_map(filtered_file, 'data/recmap.html')


salmomiks.make_fish_speed(filtered_file, fish_file)
salmomiks.make_fish_corr(filtered_file)
salmomiks.make_fish_dist(filtered_file)
