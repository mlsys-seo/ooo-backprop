import argparse
import sys 
import os
import tensorflow.compat.v1 as tf
import time

FLAGS=None


def main(_):
    worker_hosts = FLAGS.worker_hosts.split(",")
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    cluster = tf.train.ClusterSpec({"worker": worker_hosts})

    server = tf.distribute.Server(cluster, job_name='worker', task_index=FLAGS.task_index, config=config)
    print( "task index ", FLAGS.task_index )
    print( "going sleep..." )
    server.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
