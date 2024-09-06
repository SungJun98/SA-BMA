"""General interface script."""
import time

import datetime
import priorBox


args = priorBox.options()



if __name__ == "__main__":
    start_time = time.time()

    if 'prior' in args.job:
        prior = priorBox.Prior(args)
        prior.learn_prior()
        train_posterior_time = time.time()
    elif 'bayesian' in args.job:
        bayesian_learner = priorBox.bayesian_learner(args)
        bayesian_learner.learn()
        bayesian_leraning_time = time.time()


    end_time = time.time()


    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Computations time: {str(datetime.timedelta(seconds=end_time - start_time))}')
    print('-------------Job finished.-------------------------')
