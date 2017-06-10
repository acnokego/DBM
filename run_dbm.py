from dbm import DBM
import argparse
import pickle
import matplotlib.pyplot as plt



def plot_samples(data):

    plt.scatter(data[:,0], data[:,1])
    plt.savefig('./samples.png', format='png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gauss', type=int, default=2 )
    parser.add_argument('--epoch', type=int, default=50 )
    parser.add_argument('--hidden', type=int, default=8 )
    parser.add_argument('--steps', type=int, default=1 )
    parser.add_argument('--recon', type=int, default=50 )

    args = parser.parse_args()
    data_file = 'exp1_gauss'+ str(args.gauss)

    with open(data_file, 'rb') as f:
        
        samples = pickle.load(f, encoding='bytes')
        n_visible, n_hidden  = 2, args.hidden
        n_steps = args.steps
        n_epochs = args.epoch
        n_gibbs = args.recon
        
        dbm = DBM(num_visible=n_visible, num_hidden=n_hidden, CD_steps=n_steps,
                  gibb_steps=n_gibbs, num_epochs=n_epochs)

        train = samples[:8000]
        validation = samples[8000:]
        dbm.fit(train, validation)
        plot_samples(dbm.reconstruction)


