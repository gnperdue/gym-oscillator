'''
run all the tests
'''
import unittest
import argparse
import time
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--pattern', default='test', type=str,
                    help='pattern base name')
parser.add_argument('--verbosity', default=2, type=int,
                    help='test verbosity (int)')


def main(pattern, verbosity):
    run_time = int(time.time())
    logfilename = 'log_' + __file__.split('/')[-1].split('.')[0] \
        + str(run_time) + '.txt'
    logging.basicConfig(
        filename=logfilename, level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Starting...")
    LOGGER.info(__file__)

    pattern = pattern + '*.py'
    suite = unittest.TestLoader().discover('./tests/', pattern=pattern)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
