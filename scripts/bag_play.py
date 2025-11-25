#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav_msgs.msg import Odometry


class MinimalPublisher(Node):
    def __init__(self, context: rclpy.Context):
        super().__init__('minimal_publisher', context=context)
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: '%s'' % msg.data)
        self.i += 1


class MinimalSubscriber(Node):
    def __init__(self, context: rclpy.Context):
        super().__init__('minimal_subscriber', context=context)
        self.subscription = self.create_subscription(
            String, 'topic', self.listener_callback, 10
        )

    def listener_callback(self, msg):
        self.get_logger().info('I heard: '%s'' % msg.data)


def main():
    context = rclpy.context.Context()
    rclpy.init(context=context)
    minimal_publisher = MinimalPublisher(context=context)
    minimal_subscriber = MinimalSubscriber(context=context)
    executor = rclpy.executors.MultiThreadedExecutor(context=context)
    executor.add_node(minimal_publisher)
    executor.add_node(minimal_subscriber)
    try:
        executor.spin()
    except KeyboardInterrupt as e:
        print(e)
        executor.shutdown()
        minimal_publisher.destroy_node()
        minimal_subscriber.destroy_node()


if __name__ == '__main__':
    main()
