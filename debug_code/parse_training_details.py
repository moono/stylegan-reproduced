import pickle

# ######################################## #
# below will output optimizer reset period #
# ######################################## #
# lod, cur_nimg, minibatch
# 7.000, 0, 64
# 7.000, 600064, 64
# 6.000, 1200128, 64
# 6.000, 1800192, 32
# 5.000, 2400000, 32
# 5.000, 3000064, 16
# 4.000, 3600000, 16
# 4.000, 4200064, 8
# 3.000, 4800000, 8
# 3.000, 5400032, 4
# 2.000, 6000000, 4
# 2.000, 6600016, 2
# 1.000, 7200000, 2
# 1.000, 7800008, 2
# 0.000, 8400000, 2


def main():
    fn = './debug_files/training_details.pkl'

    with open(fn, 'rb') as f:
        data = pickle.load(f)

    reset_opt_list = data['reset_opt']
    cur_nimg_list = data['cur_nimg']
    lod_list = data['lod']
    minibatch_list = data['minibatch']

    len11 = len(reset_opt_list)
    len12 = len(cur_nimg_list)
    len13 = len(lod_list)
    len14 = len(minibatch_list)
    assert len11 == len12 == len13 == len14

    print('lod, cur_nimg, minibatch')
    for reset_opt, cur_nimg, lod, minibatch in zip(reset_opt_list, cur_nimg_list, lod_list, minibatch_list):
        if reset_opt:
            print('{:.03f}, {}, {}'.format(lod, cur_nimg, minibatch))
    return


if __name__ == '__main__':
    main()
