#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
/***************************************************************************
 * 
 * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
/**
 * @file test_pipeline.py
 * @author wanglong(com@baidu.com)
 * @date 2018/04/03 16:29:10
 * @brief 
 *  
 **/
"""

import unittest
import os
import paddle.reader.Pipeline as Pipeline


def make_generator(num=10):
    """ make a generator
    """

    def num_generator():
        for i in xrange(num):
            yield i

    return num_generator


class PipelineTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_batch(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        batch_size = 2
        reader = p.batch(batch_size).reader()
        for r in reader():
            assert len(r) == batch_size, 'failed to pass Pipeline.batch'

    def test_transform(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        batch_size = 2
        reader = p.batch(batch_size).transform(make_generator(20))
        for r in reader():
            assert len(r) == batch_size, 'failed to pass Pipeline.transform'

    def test_map(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        reader = p.map(lambda r: 2 * r).reader()
        i = 0
        for r in reader():
            assert i * 2 == r, 'failed to pass Pipeline.map'
            i += 1

    def test_shuffle(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        step_size = 4
        reader = p.shuffle(step_size).reader()
        i = 0
        begin = 0
        end = 0
        for r in reader():
            if i % step_size == 0:
                begin = end
                end += step_size

            assert begin <= r and r <= end, "faield to pass Pipeline.shuffle"
            i += 1

    def test_buffered(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        reader = p.buffered(3).reader()
        i = 0
        for r in reader():
            assert r == i, 'faield to pass Pipeline.buffered'
            i += 1

    def test_xmap(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        scale = 1
        func = lambda r: r * scale
        reader = p.xmap(func, 1, 4, False).reader()
        i = 0
        s = 0
        for r in reader():
            assert r == scale * i, 'faield to pass Pipeline.xmap'
            i += 1
            s += r

        p.reset()
        scale = 2
        func = lambda r: r * scale
        reader = p.xmap(func, 2, 4, False).reader()
        s2 = 0
        for r in reader():
            s2 += r

        assert s * scale == s2, 'faield to pass Pipeline.xmap with multipleprocess'

        p.reset()
        scale = 2
        func = lambda r: r * scale
        reader = p.xmap([func, func], 2, 4, False).reader()
        s3 = 0

        for r in reader():
            s3 += r

        assert scale * scale * s == s3, \
                'faield to pass Pipeline.xmap with multipleprocess and 2 maps'

    def test_combination(self):
        reader = make_generator(10)
        p = Pipeline(reader)

        scale = 2
        func = lambda r: scale * r
        batch_size = 2
        reader = p.map(func).batch(batch_size).reader()
        i = 0
        for r in reader():
            assert r[0] == scale*i and r[1] == scale*(i + 1), \
                    'failed to pass map and batch'
            i += batch_size


if __name__ == "__main__":
    unittest.main()

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
