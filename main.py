#Main file for analysis
import salmomiks
import glob

filtered_file = 'data/filtered.csv'
tag_file = 'data/moray_tag_info.csv'
rec_file_better = 'data/moray_tag_info_more.csv'
fish_file = 'data/fish_moove.csv'
rec_groups_dir = 'data/rec_groups/'
main_map_file ='data/recmap.html'
pairs_map_file ='data/pairs_recmap.html'

salmomiks.load_data('data/recievers.csv'
                    , tag_file
                    , glob.glob('data/observations/*csv')
                    , filtered_file
                    , rec_file_better
)

#salmomiks.make_rec_map(filtered_file, main_map_file)
#
#
#salmomiks.make_fish_corr(filtered_file)
#salmomiks.make_fish_dist(filtered_file)
#salmomiks.make_fish_speed(filtered_file, fish_file)
#salmomiks.check_fish_coocurrence(fish_file, rec_groups_dir)

#for each receiver print pairs of fish seen 10 min form each other
#10 min is 0.2 epoch ?
#salmomiks.check_fish_pairs(fish_file)

#fish_file = 'data_badger_thresh2epoch/data_thresh2epoch/fish_moove.csv'
#salmomiks.check_fish_pairs(fish_file, rec_file_better, pairs_map_file)
