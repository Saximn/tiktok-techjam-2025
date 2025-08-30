import { Device } from 'mediasoup-client';
import { Socket } from 'socket.io-client';

export interface MediasoupTransportOptions {
  id: string;
  iceParameters: any;
  iceCandidates: any[];
  dtlsParameters: any;
}

export interface ConsumerOptions {
  id: string;
  producerId: string;
  kind: 'audio' | 'video';
  rtpParameters: any;
}

export class MediasoupClient {
  private device?: Device;
  private sfuSocket?: Socket;
  private producerTransport?: any;
  private consumerTransport?: any;
  private producers = new Map<string, any>();
  private consumers = new Map<string, any>();
  private roomId: string;

  constructor(roomId: string) {
    this.roomId = roomId;
  }

  async initialize(sfuSocket: Socket): Promise<void> {
    this.sfuSocket = sfuSocket;
    this.device = new Device();

    // Get router RTP capabilities from SFU server
    const rtpCapabilities = await this.request('get-router-rtp-capabilities', {});
    
    // Load device with router RTP capabilities
    await this.device.load({ routerRtpCapabilities: rtpCapabilities });
    
    console.log('Mediasoup device initialized');
  }

  async createProducerTransport(): Promise<void> {
    if (!this.device || !this.sfuSocket) {
      throw new Error('Device not initialized');
    }

    const transportOptions = await this.request('create-producer-transport', {
      roomId: this.roomId
    });

    this.producerTransport = this.device.createSendTransport(transportOptions);

    // Handle transport events
    this.producerTransport.on('connect', async ({ dtlsParameters }: any, callback: any, errback: any) => {
      try {
        await this.request('connect-transport', {
          transportId: this.producerTransport.id,
          dtlsParameters
        });
        callback();
      } catch (error) {
        errback(error);
      }
    });

    this.producerTransport.on('produce', async ({ kind, rtpParameters }: any, callback: any, errback: any) => {
      try {
        const { producerId } = await this.request('produce', {
          roomId: this.roomId,
          transportId: this.producerTransport.id,
          kind,
          rtpParameters
        });
        callback({ id: producerId });
      } catch (error) {
        errback(error);
      }
    });

    this.producerTransport.on('connectionstatechange', (state: any) => {
      console.log('Producer transport state:', state);
      if (state === 'closed' || state === 'failed' || state === 'disconnected') {
        console.log('Producer transport closed');
      }
    });
  }

  async createConsumerTransport(): Promise<void> {
    if (!this.device || !this.sfuSocket) {
      throw new Error('Device not initialized');
    }

    console.log('🟢 [MEDIASOUP-CLIENT] Requesting consumer transport creation for room:', this.roomId);
    let transportOptions;
    try {
      transportOptions = await this.request('create-consumer-transport', {
        roomId: this.roomId
      });
      console.log('🟢 [MEDIASOUP-CLIENT] Consumer transport options received:', JSON.stringify(transportOptions, null, 2));
    } catch (requestError) {
      console.error('🟢 [MEDIASOUP-CLIENT] Failed to request consumer transport:', requestError);
      throw requestError;
    }

    this.consumerTransport = this.device.createRecvTransport(transportOptions);

    // Handle transport events
    this.consumerTransport.on('connect', async ({ dtlsParameters }: any, callback: any, errback: any) => {
      try {
        await this.request('connect-transport', {
          transportId: this.consumerTransport.id,
          dtlsParameters
        });
        callback();
      } catch (error) {
        errback(error);
      }
    });

    this.consumerTransport.on('connectionstatechange', (state: any) => {
      console.log('Consumer transport state:', state);
      if (state === 'closed' || state === 'failed' || state === 'disconnected') {
        console.log('Consumer transport closed');
      }
    });
  }

  hasConsumerTransport(): boolean {
    return !!this.consumerTransport;
  }

  async produce(stream: MediaStream): Promise<void> {
    if (!this.producerTransport) {
      throw new Error('Producer transport not created');
    }

    const tracks = stream.getTracks();
    
    for (const track of tracks) {
      try {
        const producer = await this.producerTransport.produce({ track });
        this.producers.set(track.kind, producer);
        
        producer.on('transportclose', () => {
          console.log(`Producer transport closed: ${producer.id}`);
        });

        producer.on('trackended', () => {
          console.log(`Producer track ended: ${producer.id}`);
        });

        console.log(`Producer created: ${producer.id} (${track.kind})`);
      } catch (error) {
        console.error(`Error creating producer for ${track.kind}:`, error);
      }
    }
  }

  async consume(producerId: string, kind: 'audio' | 'video'): Promise<MediaStream | null> {
    if (!this.consumerTransport || !this.device) {
      console.error('🔴 [MEDIASOUP-CLIENT] consume() failed - missing transport or device:', {
        hasTransport: !!this.consumerTransport,
        hasDevice: !!this.device,
        transportState: this.consumerTransport?.connectionState
      });
      throw new Error('Consumer transport or device not ready');
    }

    console.log('🟢 [MEDIASOUP-CLIENT] consume() attempting to consume:', {
      producerId,
      kind,
      transportState: this.consumerTransport.connectionState,
      deviceReady: this.device.loaded
    });

    try {
      const response = await this.request('consume', {
        roomId: this.roomId,
        transportId: this.consumerTransport.id,
        producerId,
        rtpCapabilities: this.device.rtpCapabilities
      });

      const consumerOptions = response.consumerOptions;
      const consumer = await this.consumerTransport.consume(consumerOptions);
      this.consumers.set(consumer.id, consumer);

      // Resume the consumer
      await this.request('resume-consumer', {
        consumerId: consumer.id
      });

      consumer.on('transportclose', () => {
        console.log(`Consumer transport closed: ${consumer.id}`);
      });

      consumer.on('producerclose', () => {
        console.log(`Consumer producer closed: ${consumer.id}`);
      });

      // Create media stream from consumer track
      const stream = new MediaStream();
      stream.addTrack(consumer.track);

      console.log(`Consumer created: ${consumer.id} (${kind})`);
      return stream;
    } catch (error) {
      console.error(`Error consuming ${kind}:`, error);
      return null;
    }
  }

  async stopProducing(): Promise<void> {
    // Close all producers
    for (const producer of this.producers.values()) {
      producer.close();
    }
    this.producers.clear();

    // Close producer transport
    if (this.producerTransport) {
      this.producerTransport.close();
      this.producerTransport = null;
    }
  }

  async stopConsuming(): Promise<void> {
    // Close all consumers
    for (const consumer of this.consumers.values()) {
      consumer.close();
    }
    this.consumers.clear();

    // Close consumer transport
    if (this.consumerTransport) {
      this.consumerTransport.close();
      this.consumerTransport = null;
    }
  }

  isProducerReady(): boolean {
    return !!this.producerTransport && this.producers.size > 0;
  }

  isConsumerReady(): boolean {
    return !!this.consumerTransport;
  }

  getProducers(): Map<string, any> {
    return this.producers;
  }

  getConsumers(): Map<string, any> {
    return this.consumers;
  }

  private async request(method: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.sfuSocket) {
        reject(new Error('SFU socket not connected'));
        return;
      }

      this.sfuSocket.emit(method, data, (response: any) => {
        if (response.success) {
          // Return the appropriate data based on the response structure
          if (response.rtpCapabilities) {
            resolve(response.rtpCapabilities);
          } else if (response.transportOptions) {
            resolve(response.transportOptions);
          } else if (response.consumerOptions) {
            resolve(response);
          } else if (response.producerId) {
            resolve({ producerId: response.producerId });
          } else {
            resolve(response);
          }
        } else {
          reject(new Error(response.error || 'Request failed'));
        }
      });
    });
  }
}